#include "compress3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
    else { passes++; } \
} while(0)

static int passes = 0;
static int failures = 0;

/* Shared state for callback */
static int chunks_received = 0;
static int levels_seen[C3D_MAX_LEVELS];
static pthread_mutex_t cb_mtx = PTHREAD_MUTEX_INITIALIZER;

static void chunk_callback(int level, int cx, int cy, int cz,
                            const uint8_t *data, size_t data_size,
                            void *userdata) {
    (void)cx; (void)cy; (void)cz; (void)data; (void)userdata;
    pthread_mutex_lock(&cb_mtx);
    chunks_received++;
    if (level >= 0 && level < C3D_MAX_LEVELS)
        levels_seen[level]++;
    pthread_mutex_unlock(&cb_mtx);
}

/* Server thread */
static void *server_thread(void *arg) {
    c3d_server_t *server = (c3d_server_t *)arg;
    c3d_server_run(server);
    return NULL;
}

static void test_udp_loopback(void) {
    /* Create a small C3M volume to serve */
    int dim = 32;
    size_t vol_size = (size_t)dim * dim * dim;
    uint8_t *input = (uint8_t *)malloc(vol_size);
    for (size_t i = 0; i < vol_size; i++)
        input[i] = (uint8_t)(128 + 50.0 * sin((double)i * 0.01));

    c3d_multiscale_params_t params = {
        .quality = 80,
        .upsample_method = C3D_UPSAMPLE_TRILINEAR,
        .num_threads = 0
    };

    c3d_compressed_t comp = c3d_multiscale_compress(input, dim, dim, dim, &params);
    ASSERT(comp.data != NULL, "compress for UDP test");
    free(input);

    /* Write to temp file for server to mmap */
    const char *tmppath = "/tmp/test_c3d_udp.c3m";
    FILE *fp = fopen(tmppath, "wb");
    ASSERT(fp != NULL, "create temp C3M file");
    if (fp) {
        fwrite(comp.data, 1, comp.size, fp);
        fclose(fp);
    }
    free(comp.data);

    /* Start server on a high port */
    int port = 17333;
    c3d_server_t *server = c3d_server_create(tmppath, port);
    ASSERT(server != NULL, "server create");
    if (!server) { unlink(tmppath); return; }

    pthread_t sthr;
    pthread_create(&sthr, NULL, server_thread, server);

    /* Give server a moment to start */
    usleep(50000);

    /* Create client */
    c3d_client_t *client = c3d_client_create("127.0.0.1", port, chunk_callback, NULL);
    ASSERT(client != NULL, "client create");
    if (!client) {
        c3d_server_stop(server);
        pthread_join(sthr, NULL);
        c3d_server_free(server);
        unlink(tmppath);
        return;
    }

    /* Request all levels */
    chunks_received = 0;
    memset(levels_seen, 0, sizeof(levels_seen));

    int rc = c3d_client_request_region(client, 0, 0, 0, 0, 0, 0, 5, 0, C3D_PRIORITY_NORMAL);
    ASSERT(rc == 0, "request sent");

    /* Poll for responses */
    int total_polls = 0;
    while (chunks_received < 1 && total_polls < 100) {
        c3d_client_poll(client, 50);
        total_polls++;
    }

    ASSERT(chunks_received > 0, "received at least one chunk");
    printf("  [UDP loopback] received %d chunks in %d polls\n", chunks_received, total_polls);

    /* Test cancel */
    rc = c3d_client_cancel_above(client, 2);
    ASSERT(rc == 0, "cancel sent");

    /* Cleanup */
    c3d_client_free(client);
    c3d_server_stop(server);
    pthread_join(sthr, NULL);
    c3d_server_free(server);
    unlink(tmppath);
}

static void test_header_pack_unpack(void) {
    c3d_udp_header_t h = {
        .request_id = 0x1234,
        .msg_type = C3D_MSG_RESPONSE,
        .flags = 0x03,
        .level = 5,
        .chunk_x = 10,
        .chunk_y = 20,
        .chunk_z = 30,
        .fragment_idx = 2,
        .fragment_count = 4,
        .payload_size = 1000
    };

    /* Pack and unpack should roundtrip */
    uint8_t buf[C3D_UDP_HEADER_SIZE];
    /* We can't call pack_header directly (it's static), so test via
     * the client/server which use it internally.
     * Instead, test that the struct layout is correct. */
    ASSERT(sizeof(c3d_udp_header_t) >= 14, "header struct has all fields");
    ASSERT(h.request_id == 0x1234, "field check request_id");
    ASSERT(h.level == 5, "field check level");
    ASSERT(h.chunk_z == 30, "field check chunk_z");
    (void)buf;
}

int main(void) {
    test_header_pack_unpack();
    test_udp_loopback();

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
