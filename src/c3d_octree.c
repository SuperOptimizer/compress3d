/* c3d_octree.c — Octree-adaptive volumetric compression.
 *
 * BVH-style spatial indexing: recursively subdivides the volume into an
 * octree. Uniform/empty regions are stored as single values (~1 byte).
 * Dense regions are compressed with DCT at their native leaf size.
 *
 * Format: "C3O\x01" header + serialized octree
 *
 * Node encoding (serialized depth-first):
 *   byte 0: node type
 *     0x00 = UNIFORM (1 byte value follows)
 *     0x01 = LEAF    (compressed block follows: 4-byte size + data)
 *     0x02 = BRANCH  (8 children follow recursively)
 *     0x03 = ZERO    (all zeros, no data)
 */

#include "compress3d.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define C3O_MAGIC_0 'C'
#define C3O_MAGIC_1 '3'
#define C3O_MAGIC_2 'O'
#define C3O_MAGIC_3 0x01
#define C3O_HEADER_SIZE 16

#define NODE_UNIFORM 0x00
#define NODE_LEAF    0x01
#define NODE_BRANCH  0x02
#define NODE_ZERO    0x03

/* Dynamic buffer for building the compressed output */
typedef struct {
    uint8_t *data;
    size_t size;
    size_t cap;
} obuf_t;

static void obuf_init(obuf_t *b) {
    b->cap = 65536;
    b->data = (uint8_t *)malloc(b->cap);
    b->size = 0;
}

static void obuf_ensure(obuf_t *b, size_t extra) {
    while (b->size + extra > b->cap) {
        b->cap *= 2;
        b->data = (uint8_t *)realloc(b->data, b->cap);
    }
}

static void obuf_write(obuf_t *b, const void *data, size_t len) {
    obuf_ensure(b, len);
    memcpy(b->data + b->size, data, len);
    b->size += len;
}

static void obuf_write_u8(obuf_t *b, uint8_t v) {
    obuf_ensure(b, 1);
    b->data[b->size++] = v;
}

static void obuf_write_u32(obuf_t *b, uint32_t v) {
    uint8_t buf[4] = {(uint8_t)v, (uint8_t)(v>>8), (uint8_t)(v>>16), (uint8_t)(v>>24)};
    obuf_write(b, buf, 4);
}

static uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}

/* Compute statistics of a sub-volume region */
static void region_stats(const uint8_t *vol, int vsx, int vsy,
                          int ox, int oy, int oz, int sx, int sy, int sz,
                          double *mean_out, double *stddev_out,
                          int *min_out, int *max_out, int *all_zero) {
    double sum = 0, sum2 = 0;
    int vmin = 255, vmax = 0;
    int nz = 0;
    size_t count = (size_t)sx * sy * sz;

    for (int z = 0; z < sz; z++)
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++) {
                uint8_t v = vol[(oz+z)*vsy*vsx + (oy+y)*vsx + (ox+x)];
                sum += v;
                sum2 += (double)v * v;
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
                if (v == 0) nz++;
            }

    double mean = sum / count;
    double var = sum2 / count - mean * mean;
    if (var < 0) var = 0;

    *mean_out = mean;
    *stddev_out = sqrt(var);
    *min_out = vmin;
    *max_out = vmax;
    *all_zero = (nz == (int)count);
}

/* Extract a sub-volume into a contiguous buffer */
static void extract_region(const uint8_t *vol, int vsx, int vsy,
                            int ox, int oy, int oz, int sx, int sy, int sz,
                            uint8_t *out) {
    for (int z = 0; z < sz; z++)
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
                out[z*sy*sx + y*sx + x] = vol[(oz+z)*vsy*vsx + (oy+y)*vsx + (ox+x)];
}

/* Write a sub-volume back from contiguous buffer */
static void scatter_region(const uint8_t *in, uint8_t *vol, int vsx, int vsy,
                            int ox, int oy, int oz, int sx, int sy, int sz) {
    for (int z = 0; z < sz; z++)
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
                vol[(oz+z)*vsy*vsx + (oy+y)*vsx + (ox+x)] = in[z*sy*sx + y*sx + x];
}

/* Compress a leaf region (may not be 32³ — pad to nearest power of 2) */
static void compress_leaf(const uint8_t *vol, int vsx, int vsy,
                           int ox, int oy, int oz, int sx, int sy, int sz,
                           const c3d_octree_params_t *params, obuf_t *out) {
    size_t count = (size_t)sx * sy * sz;

    /* For small regions, just store raw */
    if (count <= 64) {
        obuf_write_u8(out, NODE_LEAF);
        obuf_write_u32(out, (uint32_t)count);
        uint8_t *buf = (uint8_t *)malloc(count);
        extract_region(vol, vsx, vsy, ox, oy, oz, sx, sy, sz, buf);
        obuf_write(out, buf, count);
        free(buf);
        return;
    }

    /* Pad to 32³ and use C3D compression */
    int pad = 32;
    /* Use smaller pad if region is small */
    if (sx <= 16 && sy <= 16 && sz <= 16) pad = 16;
    if (sx <= 8 && sy <= 8 && sz <= 8) pad = 8;
    if (sx <= 4 && sy <= 4 && sz <= 4) pad = 4;

    /* Only use C3D for 32³ blocks — for non-32 sizes, store with simple prediction */
    if (sx == 32 && sy == 32 && sz == 32) {
        uint8_t block[32*32*32];
        extract_region(vol, vsx, vsy, ox, oy, oz, 32, 32, 32, block);

        float step = params->step;
        if (step <= 0) step = (params->quality >= 101) ? 0 : (50.0f * powf(0.01f, (params->quality - 1) / 99.0f));
        int max_coeffs = 0;

        c3d_compressed_t comp;
        if (params->quality >= 101)
            comp = c3d_compress(block, 101);
        else if (step > 0)
            comp = c3d_compress_step(block, step, max_coeffs);
        else
            comp = c3d_compress(block, params->quality);

        if (comp.data) {
            obuf_write_u8(out, NODE_LEAF);
            obuf_write_u32(out, (uint32_t)comp.size);
            obuf_write(out, comp.data, comp.size);
            free(comp.data);
            return;
        }
    }

    /* Fallback: store raw with delta prediction */
    uint8_t *buf = (uint8_t *)malloc(count);
    extract_region(vol, vsx, vsy, ox, oy, oz, sx, sy, sz, buf);

    /* Simple delta prediction: each voxel minus its left neighbor */
    uint8_t *delta = (uint8_t *)malloc(count);
    delta[0] = buf[0];
    for (size_t i = 1; i < count; i++)
        delta[i] = buf[i] - buf[i-1];

    obuf_write_u8(out, NODE_LEAF);
    obuf_write_u32(out, (uint32_t)(count | 0x80000000u)); /* high bit = delta-coded raw */
    obuf_write(out, delta, count);
    free(buf);
    free(delta);
}

/* Recursive octree compression */
static void octree_compress_recursive(const uint8_t *vol, int vsx, int vsy, int vsz,
                                       int ox, int oy, int oz,
                                       int sx, int sy, int sz,
                                       const c3d_octree_params_t *params,
                                       obuf_t *out, int depth) {
    /* Compute region statistics */
    double mean, stddev;
    int vmin, vmax, all_zero;
    region_stats(vol, vsx, vsy, ox, oy, oz, sx, sy, sz, &mean, &stddev, &vmin, &vmax, &all_zero);

    /* ZERO: all voxels are 0 */
    if (all_zero) {
        obuf_write_u8(out, NODE_ZERO);
        return;
    }

    /* UNIFORM: stddev below threshold */
    if (stddev <= params->uniform_threshold) {
        obuf_write_u8(out, NODE_UNIFORM);
        obuf_write_u8(out, (uint8_t)(mean + 0.5));
        return;
    }

    /* Check if we should make this a leaf (too small to split further) */
    int min_dim = params->min_leaf_dim;
    if (min_dim < 2) min_dim = 2;

    int can_split_x = (sx >= min_dim * 2);
    int can_split_y = (sy >= min_dim * 2);
    int can_split_z = (sz >= min_dim * 2);

    if (!can_split_x && !can_split_y && !can_split_z) {
        /* Leaf node — compress this region */
        compress_leaf(vol, vsx, vsy, ox, oy, oz, sx, sy, sz, params, out);
        return;
    }

    /* BRANCH: split into up to 8 children */
    obuf_write_u8(out, NODE_BRANCH);

    /* Encode which axes are split (3 bits) */
    uint8_t split_mask = 0;
    if (can_split_x) split_mask |= 0x01;
    if (can_split_y) split_mask |= 0x02;
    if (can_split_z) split_mask |= 0x04;
    obuf_write_u8(out, split_mask);

    int hx = can_split_x ? sx / 2 : sx;
    int hy = can_split_y ? sy / 2 : sy;
    int hz = can_split_z ? sz / 2 : sz;

    int nchildren_x = can_split_x ? 2 : 1;
    int nchildren_y = can_split_y ? 2 : 1;
    int nchildren_z = can_split_z ? 2 : 1;

    for (int cz = 0; cz < nchildren_z; cz++)
        for (int cy = 0; cy < nchildren_y; cy++)
            for (int cx = 0; cx < nchildren_x; cx++) {
                int child_ox = ox + cx * hx;
                int child_oy = oy + cy * hy;
                int child_oz = oz + cz * hz;
                int child_sx = (cx == nchildren_x - 1) ? (sx - cx * hx) : hx;
                int child_sy = (cy == nchildren_y - 1) ? (sy - cy * hy) : hy;
                int child_sz = (cz == nchildren_z - 1) ? (sz - cz * hz) : hz;

                octree_compress_recursive(vol, vsx, vsy, vsz,
                    child_ox, child_oy, child_oz,
                    child_sx, child_sy, child_sz,
                    params, out, depth + 1);
            }
}

c3d_compressed_t c3d_octree_compress(const uint8_t *input,
    int sx, int sy, int sz, const c3d_octree_params_t *params)
{
    c3d_compressed_t fail = { .data = NULL, .size = 0 };
    if (!input || !params || sx < 1 || sy < 1 || sz < 1) return fail;

    c3d_octree_params_t p = *params;
    if (p.uniform_threshold < 0) p.uniform_threshold = 2.0f;
    if (p.min_leaf_dim < 2) p.min_leaf_dim = 4;
    if (p.quality < 1) p.quality = 80;

    obuf_t out;
    obuf_init(&out);

    /* Write header */
    uint8_t hdr[C3O_HEADER_SIZE] = {0};
    hdr[0] = C3O_MAGIC_0; hdr[1] = C3O_MAGIC_1;
    hdr[2] = C3O_MAGIC_2; hdr[3] = C3O_MAGIC_3;
    hdr[4] = (uint8_t)(sx); hdr[5] = (uint8_t)(sx >> 8);
    hdr[6] = (uint8_t)(sy); hdr[7] = (uint8_t)(sy >> 8);
    hdr[8] = (uint8_t)(sz); hdr[9] = (uint8_t)(sz >> 8);
    hdr[10] = (uint8_t)p.quality;
    /* bytes 11-15 reserved */
    obuf_write(&out, hdr, C3O_HEADER_SIZE);

    /* Recursive octree compression */
    octree_compress_recursive(input, sx, sy, sz, 0, 0, 0, sx, sy, sz, &p, &out, 0);

    out.data = (uint8_t *)realloc(out.data, out.size);
    return (c3d_compressed_t){ .data = out.data, .size = out.size };
}

/* ── Recursive decompression ── */

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t pos;
} ibuf_t;

static uint8_t ibuf_read_u8(ibuf_t *b) {
    if (b->pos >= b->size) return 0;
    return b->data[b->pos++];
}

static uint32_t ibuf_read_u32(ibuf_t *b) {
    if (b->pos + 4 > b->size) return 0;
    uint32_t v = read_u32(b->data + b->pos);
    b->pos += 4;
    return v;
}

static int octree_decompress_recursive(ibuf_t *in, uint8_t *vol, int vsx, int vsy,
                                         int ox, int oy, int oz,
                                         int sx, int sy, int sz) {
    if (in->pos >= in->size) return -1;

    uint8_t type = ibuf_read_u8(in);

    switch (type) {
    case NODE_ZERO:
        /* Fill region with zeros */
        for (int z = 0; z < sz; z++)
            for (int y = 0; y < sy; y++)
                for (int x = 0; x < sx; x++)
                    vol[(oz+z)*vsy*vsx + (oy+y)*vsx + (ox+x)] = 0;
        return 0;

    case NODE_UNIFORM: {
        uint8_t val = ibuf_read_u8(in);
        for (int z = 0; z < sz; z++)
            for (int y = 0; y < sy; y++)
                for (int x = 0; x < sx; x++)
                    vol[(oz+z)*vsy*vsx + (oy+y)*vsx + (ox+x)] = val;
        return 0;
    }

    case NODE_LEAF: {
        uint32_t raw_size = ibuf_read_u32(in);
        int is_delta = (raw_size & 0x80000000u) != 0;
        uint32_t data_size = raw_size & 0x7FFFFFFFu;

        if (in->pos + data_size > in->size) return -1;

        size_t count = (size_t)sx * sy * sz;

        if (is_delta) {
            /* Delta-coded raw data */
            if (data_size != (uint32_t)count) return -1;
            uint8_t *delta = (uint8_t *)malloc(count);
            memcpy(delta, in->data + in->pos, count);
            in->pos += count;

            /* Undo delta prediction */
            uint8_t *buf = (uint8_t *)malloc(count);
            buf[0] = delta[0];
            for (size_t i = 1; i < count; i++)
                buf[i] = buf[i-1] + delta[i];

            scatter_region(buf, vol, vsx, vsy, ox, oy, oz, sx, sy, sz);
            free(delta); free(buf);
        } else if (sx == 32 && sy == 32 && sz == 32 && data_size < count) {
            /* C3D compressed 32³ block */
            uint8_t block[32*32*32];
            if (c3d_decompress(in->data + in->pos, data_size, block) != 0) {
                in->pos += data_size;
                return -1;
            }
            in->pos += data_size;
            scatter_region(block, vol, vsx, vsy, ox, oy, oz, 32, 32, 32);
        } else {
            /* Raw data */
            if (data_size != (uint32_t)count && data_size < count) return -1;
            uint8_t *buf = (uint8_t *)malloc(count);
            memcpy(buf, in->data + in->pos, count > data_size ? data_size : count);
            in->pos += data_size;
            scatter_region(buf, vol, vsx, vsy, ox, oy, oz, sx, sy, sz);
            free(buf);
        }
        return 0;
    }

    case NODE_BRANCH: {
        uint8_t split_mask = ibuf_read_u8(in);
        int split_x = (split_mask & 0x01) != 0;
        int split_y = (split_mask & 0x02) != 0;
        int split_z = (split_mask & 0x04) != 0;

        int hx = split_x ? sx / 2 : sx;
        int hy = split_y ? sy / 2 : sy;
        int hz = split_z ? sz / 2 : sz;

        int nx = split_x ? 2 : 1;
        int ny = split_y ? 2 : 1;
        int nz = split_z ? 2 : 1;

        for (int cz = 0; cz < nz; cz++)
            for (int cy = 0; cy < ny; cy++)
                for (int cx = 0; cx < nx; cx++) {
                    int child_ox = ox + cx * hx;
                    int child_oy = oy + cy * hy;
                    int child_oz = oz + cz * hz;
                    int child_sx = (cx == nx - 1) ? (sx - cx * hx) : hx;
                    int child_sy = (cy == ny - 1) ? (sy - cy * hy) : hy;
                    int child_sz = (cz == nz - 1) ? (sz - cz * hz) : hz;

                    int rc = octree_decompress_recursive(in, vol, vsx, vsy,
                        child_ox, child_oy, child_oz,
                        child_sx, child_sy, child_sz);
                    if (rc != 0) return rc;
                }
        return 0;
    }

    default:
        return -1;
    }
}

int c3d_octree_decompress(const uint8_t *compressed, size_t compressed_size,
                           uint8_t *output) {
    if (!compressed || !output || compressed_size < C3O_HEADER_SIZE) return -1;
    if (compressed[0] != C3O_MAGIC_0 || compressed[1] != C3O_MAGIC_1 ||
        compressed[2] != C3O_MAGIC_2 || compressed[3] != C3O_MAGIC_3)
        return -1;

    int sx = (int)compressed[4] | ((int)compressed[5] << 8);
    int sy = (int)compressed[6] | ((int)compressed[7] << 8);
    int sz = (int)compressed[8] | ((int)compressed[9] << 8);

    memset(output, 0, (size_t)sx * sy * sz);

    ibuf_t in = { .data = compressed, .size = compressed_size, .pos = C3O_HEADER_SIZE };
    return octree_decompress_recursive(&in, output, sx, sy, 0, 0, 0, sx, sy, sz);
}

int c3d_octree_get_dims(const uint8_t *compressed, size_t compressed_size,
                         int *sx, int *sy, int *sz) {
    if (!compressed || compressed_size < C3O_HEADER_SIZE) return -1;
    if (compressed[0] != C3O_MAGIC_0 || compressed[1] != C3O_MAGIC_1 ||
        compressed[2] != C3O_MAGIC_2 || compressed[3] != C3O_MAGIC_3)
        return -1;

    if (sx) *sx = (int)compressed[4] | ((int)compressed[5] << 8);
    if (sy) *sy = (int)compressed[6] | ((int)compressed[7] << 8);
    if (sz) *sz = (int)compressed[8] | ((int)compressed[9] << 8);
    return 0;
}

/* Statistics gathering (recursive) */
static void octree_stats_recursive(ibuf_t *in, c3d_octree_stats_t *stats, int depth) {
    if (in->pos >= in->size) return;
    size_t start = in->pos;
    uint8_t type = ibuf_read_u8(in);
    stats->total_nodes++;
    if (depth > stats->max_depth) stats->max_depth = depth;

    switch (type) {
    case NODE_ZERO:
        stats->uniform_nodes++;
        stats->uniform_bytes += (in->pos - start);
        break;
    case NODE_UNIFORM:
        ibuf_read_u8(in);
        stats->uniform_nodes++;
        stats->uniform_bytes += (in->pos - start);
        break;
    case NODE_LEAF: {
        uint32_t raw_size = ibuf_read_u32(in);
        uint32_t data_size = raw_size & 0x7FFFFFFFu;
        in->pos += data_size;
        stats->leaf_nodes++;
        stats->leaf_bytes += (in->pos - start);
        break;
    }
    case NODE_BRANCH: {
        uint8_t mask = ibuf_read_u8(in);
        stats->branch_nodes++;
        stats->tree_bytes += (in->pos - start);
        int nx = (mask & 1) ? 2 : 1;
        int ny = (mask & 2) ? 2 : 1;
        int nz = (mask & 4) ? 2 : 1;
        for (int i = 0; i < nx * ny * nz; i++)
            octree_stats_recursive(in, stats, depth + 1);
        break;
    }
    }
}

int c3d_octree_stats(const uint8_t *compressed, size_t compressed_size,
                      c3d_octree_stats_t *stats) {
    if (!compressed || !stats || compressed_size < C3O_HEADER_SIZE) return -1;
    memset(stats, 0, sizeof(*stats));
    ibuf_t in = { .data = compressed, .size = compressed_size, .pos = C3O_HEADER_SIZE };
    octree_stats_recursive(&in, stats, 0);
    return 0;
}
