/* c3d_net.c — UDP server and client for streaming C3D multiscale data.
 *
 * Protocol: coarse-to-fine progressive delivery of C3M pyramid levels.
 * Each datagram carries one compressed block (or fragment thereof).
 * Designed for low-latency volumetric data viewing (e.g., Neuroglancer).
 */

#include "compress3d.h"

#ifndef _WIN32

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <time.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

/* ══════════════════════════════════════════════════════════════════════════════
 * Wire format helpers
 * ══════════════════════════════════════════════════════════════════════════════ */

static void pack_header(uint8_t *buf, const c3d_udp_header_t *h) {
    buf[0] = (uint8_t)(h->request_id);
    buf[1] = (uint8_t)(h->request_id >> 8);
    buf[2] = h->msg_type;
    buf[3] = h->flags;
    buf[4] = h->level;
    buf[5] = (uint8_t)(h->chunk_x);
    buf[6] = (uint8_t)(h->chunk_x >> 8);
    buf[7] = (uint8_t)(h->chunk_y);
    buf[8] = (uint8_t)(h->chunk_y >> 8);
    buf[9] = (uint8_t)(h->chunk_z);
    buf[10] = (uint8_t)(h->chunk_z >> 8);
    buf[11] = h->fragment_idx;
    buf[12] = h->fragment_count;
    buf[13] = 0; /* reserved */
    buf[14] = (uint8_t)(h->payload_size);
    buf[15] = (uint8_t)(h->payload_size >> 8);
}

static void unpack_header(const uint8_t *buf, c3d_udp_header_t *h) {
    h->request_id = (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
    h->msg_type = buf[2];
    h->flags = buf[3];
    h->level = buf[4];
    h->chunk_x = (uint16_t)buf[5] | ((uint16_t)buf[6] << 8);
    h->chunk_y = (uint16_t)buf[7] | ((uint16_t)buf[8] << 8);
    h->chunk_z = (uint16_t)buf[9] | ((uint16_t)buf[10] << 8);
    h->fragment_idx = buf[11];
    h->fragment_count = buf[12];
    h->payload_size = (uint16_t)buf[14] | ((uint16_t)buf[15] << 8);
}

/* ══════════════════════════════════════════════════════════════════════════════
 * Server
 * ══════════════════════════════════════════════════════════════════════════════ */

#define MAX_CLIENTS 64
#define SEND_QUEUE_SIZE 4096

typedef struct {
    struct sockaddr_in addr;
    uint16_t last_request_id;
    int active;
} client_slot_t;

typedef struct {
    uint8_t data[C3D_UDP_HEADER_SIZE + C3D_UDP_MAX_PAYLOAD];
    int len;
    struct sockaddr_in dest;
} send_item_t;

struct c3d_server {
    int sock_fd;
    int running;
    int port;

    /* Volume data (mmap'd C3M file) */
    uint8_t *volume_data;
    size_t volume_size;
    int volume_fd;
    c3d_multiscale_header_t hdr;
    c3d_level_cache_t *cache;

    /* Client tracking */
    client_slot_t clients[MAX_CLIENTS];

    /* Send queue (ring buffer) */
    send_item_t *send_queue;
    int sq_head, sq_tail, sq_count;
    pthread_mutex_t sq_mtx;
};

c3d_server_t *c3d_server_create(const char *volume_path, int port) {
    if (!volume_path) return NULL;
    if (port <= 0) port = C3D_UDP_PORT_DEFAULT;

    /* mmap the volume file */
    int fd = open(volume_path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    size_t fsize = (size_t)st.st_size;

    uint8_t *data = (uint8_t *)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { close(fd); return NULL; }

    c3d_multiscale_header_t hdr;
    if (c3d_multiscale_header(data, fsize, &hdr) != 0) {
        munmap(data, fsize);
        close(fd);
        return NULL;
    }

    /* Create UDP socket */
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { munmap(data, fsize); close(fd); return NULL; }

    struct sockaddr_in bind_addr;
    memset(&bind_addr, 0, sizeof(bind_addr));
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = htons((uint16_t)port);
    bind_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (struct sockaddr *)&bind_addr, sizeof(bind_addr)) != 0) {
        close(sock); munmap(data, fsize); close(fd);
        return NULL;
    }

    c3d_server_t *s = (c3d_server_t *)calloc(1, sizeof(c3d_server_t));
    if (!s) { close(sock); munmap(data, fsize); close(fd); return NULL; }

    s->sock_fd = sock;
    s->port = port;
    s->volume_data = data;
    s->volume_size = fsize;
    s->volume_fd = fd;
    s->hdr = hdr;
    s->cache = c3d_level_cache_create();
    s->send_queue = (send_item_t *)calloc(SEND_QUEUE_SIZE, sizeof(send_item_t));
    pthread_mutex_init(&s->sq_mtx, NULL);

    /* Pre-decode all levels into cache for fast serving */
    for (int lev = 0; lev < hdr.num_levels; lev++) {
        size_t lsz = (size_t)hdr.levels[lev].dim_x * hdr.levels[lev].dim_y * hdr.levels[lev].dim_z;
        uint8_t *buf = (uint8_t *)malloc(lsz);
        if (buf) {
            c3d_multiscale_decompress_level(data, fsize, lev, buf, s->cache);
            free(buf);
        }
    }

    return s;
}

/* Enqueue a response datagram for sending. */
static void server_enqueue(c3d_server_t *s, const c3d_udp_header_t *hdr,
                            const uint8_t *payload, int payload_len,
                            const struct sockaddr_in *dest) {
    pthread_mutex_lock(&s->sq_mtx);
    if (s->sq_count < SEND_QUEUE_SIZE) {
        send_item_t *item = &s->send_queue[s->sq_head];
        pack_header(item->data, hdr);
        if (payload && payload_len > 0)
            memcpy(item->data + C3D_UDP_HEADER_SIZE, payload, payload_len);
        item->len = C3D_UDP_HEADER_SIZE + payload_len;
        item->dest = *dest;
        s->sq_head = (s->sq_head + 1) % SEND_QUEUE_SIZE;
        s->sq_count++;
    }
    pthread_mutex_unlock(&s->sq_mtx);
}

/* Send a block (possibly fragmented) as response datagrams. */
static void server_send_block(c3d_server_t *s, uint16_t request_id,
                               int level, int cx, int cy, int cz,
                               const uint8_t *data, size_t data_len,
                               const struct sockaddr_in *dest) {
    int nfrags = (int)((data_len + C3D_UDP_MAX_PAYLOAD - 1) / C3D_UDP_MAX_PAYLOAD);
    if (nfrags < 1) nfrags = 1;

    for (int f = 0; f < nfrags; f++) {
        size_t off = (size_t)f * C3D_UDP_MAX_PAYLOAD;
        size_t remaining = data_len - off;
        int plen = (remaining > C3D_UDP_MAX_PAYLOAD) ? C3D_UDP_MAX_PAYLOAD : (int)remaining;

        c3d_udp_header_t hdr;
        hdr.request_id = request_id;
        hdr.msg_type = C3D_MSG_RESPONSE;
        hdr.flags = (f < nfrags - 1) ? 0x01 : 0x02; /* more_fragments / last_fragment */
        hdr.level = (uint8_t)level;
        hdr.chunk_x = (uint16_t)cx;
        hdr.chunk_y = (uint16_t)cy;
        hdr.chunk_z = (uint16_t)cz;
        hdr.fragment_idx = (uint8_t)f;
        hdr.fragment_count = (uint8_t)nfrags;
        hdr.payload_size = (uint16_t)plen;

        server_enqueue(s, &hdr, data + off, plen, dest);
    }
}

/* Handle a request: decode and send the requested region coarse-to-fine. */
static void server_handle_request(c3d_server_t *s, const uint8_t *pkt, int pkt_len,
                                   const struct sockaddr_in *from) {
    if (pkt_len < C3D_UDP_HEADER_SIZE + 4) return;

    c3d_udp_header_t req;
    unpack_header(pkt, &req);
    if (req.msg_type != C3D_MSG_REQUEST) return;

    const uint8_t *payload = pkt + C3D_UDP_HEADER_SIZE;
    int region_type = payload[0];
    int max_level = payload[1];
    int min_level = payload[2];
    /* int priority = payload[3]; -- unused for now */

    if (max_level >= s->hdr.num_levels) max_level = s->hdr.num_levels - 1;
    if (min_level < 0) min_level = 0;

    /* Send levels from coarse to fine */
    for (int lev = min_level; lev <= max_level; lev++) {
        if (!s->running) break;

        /* Get the compressed level data directly from the C3M container */
        const uint8_t *lev_data = s->volume_data + s->hdr.levels[lev].offset;
        size_t lev_size = s->hdr.levels[lev].size;

        if (region_type == C3D_REGION_SINGLE) {
            /* Send raw compressed level data */
            server_send_block(s, req.request_id, lev, 0, 0, 0,
                              lev_data, lev_size, from);
        } else {
            /* Box region — for now, send the entire level.
             * TODO: spatial filtering of blocks for box/sphere regions. */
            server_send_block(s, req.request_id, lev, 0, 0, 0,
                              lev_data, lev_size, from);
        }
    }
}

/* Drain send queue — send up to `max_packets` datagrams. */
static int server_drain_queue(c3d_server_t *s, int max_packets) {
    int sent = 0;
    pthread_mutex_lock(&s->sq_mtx);
    while (s->sq_count > 0 && sent < max_packets) {
        send_item_t *item = &s->send_queue[s->sq_tail];
        sendto(s->sock_fd, item->data, item->len, 0,
               (struct sockaddr *)&item->dest, sizeof(item->dest));
        s->sq_tail = (s->sq_tail + 1) % SEND_QUEUE_SIZE;
        s->sq_count--;
        sent++;
    }
    pthread_mutex_unlock(&s->sq_mtx);
    return sent;
}

int c3d_server_run(c3d_server_t *server) {
    if (!server) return -1;
    server->running = 1;

    struct pollfd pfd;
    pfd.fd = server->sock_fd;
    pfd.events = POLLIN;

    while (server->running) {
        int rc = poll(&pfd, 1, 10); /* 10ms timeout */

        if (rc > 0 && (pfd.revents & POLLIN)) {
            uint8_t buf[2048];
            struct sockaddr_in from;
            socklen_t from_len = sizeof(from);
            ssize_t n = recvfrom(server->sock_fd, buf, sizeof(buf), 0,
                                  (struct sockaddr *)&from, &from_len);
            if (n >= C3D_UDP_HEADER_SIZE)
                server_handle_request(server, buf, (int)n, &from);
        }

        /* Drain send queue — pace at ~100 packets per poll cycle */
        server_drain_queue(server, 100);
    }

    return 0;
}

void c3d_server_stop(c3d_server_t *server) {
    if (server) server->running = 0;
}

void c3d_server_free(c3d_server_t *server) {
    if (!server) return;
    if (server->running) c3d_server_stop(server);
    close(server->sock_fd);
    if (server->volume_data)
        munmap(server->volume_data, server->volume_size);
    if (server->volume_fd >= 0)
        close(server->volume_fd);
    c3d_level_cache_free(server->cache);
    free(server->send_queue);
    pthread_mutex_destroy(&server->sq_mtx);
    free(server);
}

/* ══════════════════════════════════════════════════════════════════════════════
 * Client
 * ══════════════════════════════════════════════════════════════════════════════ */

#define FRAG_BUFS 256

typedef struct {
    uint16_t request_id;
    uint8_t  level;
    uint16_t chunk_x, chunk_y, chunk_z;
    uint8_t  fragment_count;
    uint8_t  received_mask[32]; /* up to 256 fragments */
    uint8_t *data;
    size_t   data_cap;
    size_t   total_size;
    int      complete;
} frag_assembly_t;

struct c3d_client {
    int sock_fd;
    struct sockaddr_in server_addr;
    c3d_chunk_cb callback;
    void *userdata;
    uint16_t next_request_id;

    /* Fragment reassembly */
    frag_assembly_t frags[FRAG_BUFS];
};

c3d_client_t *c3d_client_create(const char *host, int port,
                                 c3d_chunk_cb cb, void *userdata) {
    if (!host || !cb) return NULL;
    if (port <= 0) port = C3D_UDP_PORT_DEFAULT;

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return NULL;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);

    struct hostent *he = gethostbyname(host);
    if (!he) { close(sock); return NULL; }
    memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);

    c3d_client_t *c = (c3d_client_t *)calloc(1, sizeof(c3d_client_t));
    if (!c) { close(sock); return NULL; }
    c->sock_fd = sock;
    c->server_addr = addr;
    c->callback = cb;
    c->userdata = userdata;
    c->next_request_id = 1;

    return c;
}

int c3d_client_request_region(c3d_client_t *c,
    int x0, int y0, int z0, int x1, int y1, int z1,
    int max_level, int min_level, int priority)
{
    if (!c) return -1;

    uint8_t pkt[C3D_UDP_HEADER_SIZE + 16];
    memset(pkt, 0, sizeof(pkt));

    c3d_udp_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.request_id = c->next_request_id++;
    hdr.msg_type = C3D_MSG_REQUEST;
    hdr.payload_size = 16;
    pack_header(pkt, &hdr);

    uint8_t *payload = pkt + C3D_UDP_HEADER_SIZE;
    payload[0] = C3D_REGION_BOX;
    payload[1] = (uint8_t)max_level;
    payload[2] = (uint8_t)min_level;
    payload[3] = (uint8_t)priority;
    payload[4] = (uint8_t)(x0); payload[5] = (uint8_t)(x0 >> 8);
    payload[6] = (uint8_t)(y0); payload[7] = (uint8_t)(y0 >> 8);
    payload[8] = (uint8_t)(z0); payload[9] = (uint8_t)(z0 >> 8);
    payload[10] = (uint8_t)(x1); payload[11] = (uint8_t)(x1 >> 8);
    payload[12] = (uint8_t)(y1); payload[13] = (uint8_t)(y1 >> 8);
    payload[14] = (uint8_t)(z1); payload[15] = (uint8_t)(z1 >> 8);

    ssize_t sent = sendto(c->sock_fd, pkt, C3D_UDP_HEADER_SIZE + 16, 0,
                           (struct sockaddr *)&c->server_addr, sizeof(c->server_addr));
    return (sent > 0) ? 0 : -1;
}

int c3d_client_cancel_above(c3d_client_t *c, int level) {
    if (!c) return -1;

    uint8_t pkt[C3D_UDP_HEADER_SIZE + 4];
    memset(pkt, 0, sizeof(pkt));

    c3d_udp_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.request_id = c->next_request_id++;
    hdr.msg_type = C3D_MSG_CANCEL;
    hdr.level = (uint8_t)level;
    hdr.payload_size = 0;
    pack_header(pkt, &hdr);

    ssize_t sent = sendto(c->sock_fd, pkt, C3D_UDP_HEADER_SIZE, 0,
                           (struct sockaddr *)&c->server_addr, sizeof(c->server_addr));
    return (sent > 0) ? 0 : -1;
}

/* Find or allocate a fragment assembly slot. */
static frag_assembly_t *find_frag_slot(c3d_client_t *c, const c3d_udp_header_t *hdr) {
    /* Look for existing slot */
    for (int i = 0; i < FRAG_BUFS; i++) {
        frag_assembly_t *f = &c->frags[i];
        if (f->data && f->request_id == hdr->request_id &&
            f->level == hdr->level &&
            f->chunk_x == hdr->chunk_x &&
            f->chunk_y == hdr->chunk_y &&
            f->chunk_z == hdr->chunk_z)
            return f;
    }
    /* Allocate new slot */
    for (int i = 0; i < FRAG_BUFS; i++) {
        frag_assembly_t *f = &c->frags[i];
        if (!f->data || f->complete) {
            free(f->data);
            memset(f, 0, sizeof(*f));
            f->request_id = hdr->request_id;
            f->level = hdr->level;
            f->chunk_x = hdr->chunk_x;
            f->chunk_y = hdr->chunk_y;
            f->chunk_z = hdr->chunk_z;
            f->fragment_count = hdr->fragment_count;
            f->data_cap = (size_t)hdr->fragment_count * C3D_UDP_MAX_PAYLOAD;
            f->data = (uint8_t *)calloc(1, f->data_cap);
            return f;
        }
    }
    return NULL; /* all slots full */
}

int c3d_client_poll(c3d_client_t *c, int timeout_ms) {
    if (!c) return -1;

    struct pollfd pfd;
    pfd.fd = c->sock_fd;
    pfd.events = POLLIN;

    int chunks_received = 0;
    int rc = poll(&pfd, 1, timeout_ms);
    if (rc <= 0) return rc == 0 ? 0 : -1;

    /* Read all available datagrams */
    while (1) {
        uint8_t buf[2048];
        ssize_t n = recv(c->sock_fd, buf, sizeof(buf), MSG_DONTWAIT);
        if (n < C3D_UDP_HEADER_SIZE) break;

        c3d_udp_header_t hdr;
        unpack_header(buf, &hdr);

        if (hdr.msg_type != C3D_MSG_RESPONSE) continue;
        if (hdr.payload_size > (int)(n - C3D_UDP_HEADER_SIZE)) continue;

        if (hdr.fragment_count <= 1) {
            /* Single-fragment response — deliver immediately */
            c->callback(hdr.level, hdr.chunk_x, hdr.chunk_y, hdr.chunk_z,
                        buf + C3D_UDP_HEADER_SIZE, hdr.payload_size, c->userdata);
            chunks_received++;
        } else {
            /* Multi-fragment — reassemble */
            frag_assembly_t *f = find_frag_slot(c, &hdr);
            if (!f) continue;

            int idx = hdr.fragment_idx;
            if (idx >= f->fragment_count) continue;
            if (f->received_mask[idx / 8] & (1 << (idx % 8))) continue; /* dup */

            size_t off = (size_t)idx * C3D_UDP_MAX_PAYLOAD;
            if (off + hdr.payload_size <= f->data_cap) {
                memcpy(f->data + off, buf + C3D_UDP_HEADER_SIZE, hdr.payload_size);
                f->received_mask[idx / 8] |= (1 << (idx % 8));
                if (hdr.flags & 0x02) /* last fragment */
                    f->total_size = off + hdr.payload_size;
            }

            /* Check if all fragments received */
            int all_received = 1;
            for (int i = 0; i < f->fragment_count; i++) {
                if (!(f->received_mask[i / 8] & (1 << (i % 8)))) {
                    all_received = 0;
                    break;
                }
            }

            if (all_received && f->total_size > 0) {
                f->complete = 1;
                c->callback(f->level, f->chunk_x, f->chunk_y, f->chunk_z,
                            f->data, f->total_size, c->userdata);
                chunks_received++;
            }
        }
    }

    return chunks_received;
}

void c3d_client_free(c3d_client_t *c) {
    if (!c) return;
    close(c->sock_fd);
    for (int i = 0; i < FRAG_BUFS; i++)
        free(c->frags[i].data);
    free(c);
}

#endif /* !_WIN32 */
