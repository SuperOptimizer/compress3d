/* c3d_octree.c — Octree-adaptive volumetric compression.
 *
 * BVH-style spatial indexing: recursively subdivides the volume into an
 * octree. The split/leaf decision is made by rate-distortion optimization:
 * at each node, try compressing as a single leaf vs splitting into children,
 * and pick whichever is smaller at acceptable quality.
 *
 * Node types:
 *   0x00 = UNIFORM (1 byte value follows — entire region is one value)
 *   0x01 = LEAF    (4-byte size + compressed data — DCT or delta-coded)
 *   0x02 = BRANCH  (split_mask byte + children recursively)
 *   0x03 = ZERO    (entire region is all zeros)
 *
 * Leaf compression adapts to region size:
 *   32³: full C3D DCT pipeline (best quality/ratio)
 *   16³, 8³, 4³: mini-DCT with precomputed tables
 *   < 4³ or non-cubic: delta-coded raw
 *
 * The RD decision ensures non-sparse data (gradients, textures) stays
 * efficient — the octree stops splitting when a single leaf is cheaper.
 *
 * Format: "C3O\x02" header (v2) + serialized octree
 */

#include "compress3d.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define C3O_MAGIC_0 'C'
#define C3O_MAGIC_1 '3'
#define C3O_MAGIC_2 'O'
#define C3O_MAGIC_3 0x02   /* v2 format */
#define C3O_HEADER_SIZE 16

#define NODE_UNIFORM 0x00
#define NODE_LEAF    0x01
#define NODE_BRANCH  0x02
#define NODE_ZERO    0x03

/* ── Dynamic output buffer ── */
typedef struct { uint8_t *data; size_t size, cap; } obuf_t;

static void obuf_init(obuf_t *b) { b->cap = 65536; b->data = malloc(b->cap); b->size = 0; }
static void obuf_ensure(obuf_t *b, size_t n) {
    while (b->size + n > b->cap) { b->cap *= 2; b->data = realloc(b->data, b->cap); }
}
static void obuf_write(obuf_t *b, const void *d, size_t n) {
    obuf_ensure(b, n); memcpy(b->data + b->size, d, n); b->size += n;
}
static void obuf_u8(obuf_t *b, uint8_t v) { obuf_ensure(b, 1); b->data[b->size++] = v; }
static void obuf_u16(obuf_t *b, uint16_t v) {
    uint8_t buf[2] = {(uint8_t)v, (uint8_t)(v>>8)}; obuf_write(b, buf, 2);
}
static void obuf_u32(obuf_t *b, uint32_t v) {
    uint8_t buf[4] = {(uint8_t)v, (uint8_t)(v>>8), (uint8_t)(v>>16), (uint8_t)(v>>24)};
    obuf_write(b, buf, 4);
}

/* ── Dynamic input buffer ── */
typedef struct { const uint8_t *data; size_t size, pos; } ibuf_t;
static uint8_t ibuf_u8(ibuf_t *b) { return (b->pos < b->size) ? b->data[b->pos++] : 0; }
static uint16_t ibuf_u16(ibuf_t *b) {
    if (b->pos + 2 > b->size) return 0;
    uint16_t v = b->data[b->pos] | ((uint16_t)b->data[b->pos+1] << 8);
    b->pos += 2; return v;
}
static uint32_t ibuf_u32(ibuf_t *b) {
    if (b->pos + 4 > b->size) return 0;
    uint32_t v = b->data[b->pos] | ((uint32_t)b->data[b->pos+1]<<8)
               | ((uint32_t)b->data[b->pos+2]<<16) | ((uint32_t)b->data[b->pos+3]<<24);
    b->pos += 4; return v;
}

/* ── Region helpers ── */

static void region_stats(const uint8_t *vol, int vsx, int vsy,
                          int ox, int oy, int oz, int sx, int sy, int sz,
                          double *mean, double *stddev, int *all_zero) {
    double sum = 0, sum2 = 0;
    size_t count = (size_t)sx * sy * sz;
    int nz = 0;
    for (int z = 0; z < sz; z++)
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++) {
                uint8_t v = vol[(oz+z)*vsy*vsx + (oy+y)*vsx + (ox+x)];
                sum += v; sum2 += (double)v * v;
                if (v == 0) nz++;
            }
    double m = sum / count;
    double var = sum2 / count - m * m;
    *mean = m;
    *stddev = (var > 0) ? sqrt(var) : 0;
    *all_zero = (nz == (int)count);
}

static void extract_region(const uint8_t *vol, int vsx, int vsy,
                            int ox, int oy, int oz, int sx, int sy, int sz,
                            uint8_t *out) {
    for (int z = 0; z < sz; z++)
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
                out[z*sy*sx + y*sx + x] = vol[(oz+z)*vsy*vsx + (oy+y)*vsx + (ox+x)];
}

static void scatter_region(const uint8_t *in, uint8_t *vol, int vsx, int vsy,
                            int ox, int oy, int oz, int sx, int sy, int sz) {
    for (int z = 0; z < sz; z++)
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
                vol[(oz+z)*vsy*vsx + (oy+y)*vsx + (ox+x)] = in[z*sy*sx + y*sx + x];
}

/* ── Leaf compression ── */

/* Compress a region as a leaf node. Returns compressed data in a temporary buffer.
 * For 32³: uses full C3D DCT pipeline.
 * For smaller/non-cubic: uses delta coding with histogram packing. */
static c3d_compressed_t compress_leaf_data(const uint8_t *vol, int vsx, int vsy,
                                            int ox, int oy, int oz,
                                            int sx, int sy, int sz,
                                            const c3d_octree_params_t *params) {
    size_t count = (size_t)sx * sy * sz;
    c3d_compressed_t fail = { NULL, 0 };

    /* Extract contiguous buffer */
    uint8_t *buf = malloc(count);
    if (!buf) return fail;
    extract_region(vol, vsx, vsy, ox, oy, oz, sx, sy, sz, buf);

    /* Regions that are multiples of 32 on all axes: tile into 32³ C3D blocks.
     * This supports 32³, 64³, 128³, etc. as efficient DCT leaves. */
    if (sx >= 32 && sy >= 32 && sz >= 32 && (sx % 32 == 0) && (sy % 32 == 0) && (sz % 32 == 0)) {
        int bx = sx / 32, by = sy / 32, bz = sz / 32;
        int nblocks = bx * by * bz;

        /* Compress each 32³ sub-block and concatenate:
         * [nblocks(u16)] [bx(u8) by(u8) bz(u8)] [sizes(nblocks*u32)] [data...] */
        size_t bound = c3d_compress_bound();
        uint8_t **comp_data = calloc(nblocks, sizeof(uint8_t *));
        uint32_t *comp_sizes = calloc(nblocks, sizeof(uint32_t));
        if (!comp_data || !comp_sizes) { free(buf); free(comp_data); free(comp_sizes); return fail; }

        float step = params->step;
        if (step <= 0 && params->quality < 101)
            step = 50.0f * powf(0.01f, ((float)params->quality - 1) / 99.0f);

        size_t total_data = 0;
        for (int ibz = 0; ibz < bz; ibz++)
            for (int iby = 0; iby < by; iby++)
                for (int ibx = 0; ibx < bx; ibx++) {
                    int bi = ibz*by*bx + iby*bx + ibx;
                    uint8_t block[32*32*32];
                    for (int z = 0; z < 32; z++)
                        for (int y = 0; y < 32; y++)
                            memcpy(&block[z*32*32 + y*32],
                                   &buf[(ibz*32+z)*sy*sx + (iby*32+y)*sx + ibx*32], 32);

                    c3d_compressed_t c;
                    if (params->quality >= 101)
                        c = c3d_compress(block, 101);
                    else
                        c = c3d_compress_step(block, step, 0);

                    if (c.data) {
                        comp_data[bi] = c.data;
                        comp_sizes[bi] = (uint32_t)c.size;
                        total_data += c.size;
                    }
                }

        /* Pack: header(5) + sizes(nblocks*4) + data */
        size_t out_size = 5 + (size_t)nblocks * 4 + total_data;
        uint8_t *out = malloc(out_size);
        if (!out) {
            for (int i = 0; i < nblocks; i++) free(comp_data[i]);
            free(comp_data); free(comp_sizes); free(buf); return fail;
        }

        /* Write tiled header */
        out[0] = 0xFE; /* marker: tiled C3D blocks */
        out[1] = (uint8_t)bx; out[2] = (uint8_t)by; out[3] = (uint8_t)bz;
        out[4] = (uint8_t)nblocks;
        size_t pos = 5;
        for (int i = 0; i < nblocks; i++) {
            out[pos++] = (uint8_t)(comp_sizes[i]);
            out[pos++] = (uint8_t)(comp_sizes[i] >> 8);
            out[pos++] = (uint8_t)(comp_sizes[i] >> 16);
            out[pos++] = (uint8_t)(comp_sizes[i] >> 24);
        }
        for (int i = 0; i < nblocks; i++) {
            if (comp_data[i]) {
                memcpy(out + pos, comp_data[i], comp_sizes[i]);
                pos += comp_sizes[i];
                free(comp_data[i]);
            }
        }
        free(comp_data); free(comp_sizes); free(buf);
        return (c3d_compressed_t){ .data = out, .size = pos };
    }

    /* Single 32³ block (exact fit) */
    if (sx == 32 && sy == 32 && sz == 32) {
        c3d_compressed_t comp;
        if (params->quality >= 101)
            comp = c3d_compress(buf, 101);
        else {
            float step = params->step;
            if (step <= 0)
                step = 50.0f * powf(0.01f, ((float)params->quality - 1) / 99.0f);
            comp = c3d_compress_step(buf, step, 0);
        }
        free(buf);
        return comp;
    }

    /* For other sizes: delta prediction + simple encoding.
     * Predict each voxel from its left neighbor (in linearized order).
     * Then histogram-pack the residuals and store compactly. */

    /* Step 1: compute prediction residuals */
    uint8_t *residuals = malloc(count);
    if (!residuals) { free(buf); return fail; }

    /* 3D causal prediction: predict from left, above, and behind neighbors */
    for (int z = 0; z < sz; z++)
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++) {
                int idx = z*sy*sx + y*sx + x;
                int pred = 0;
                int npred = 0;
                if (x > 0) { pred += buf[z*sy*sx + y*sx + (x-1)]; npred++; }
                if (y > 0) { pred += buf[z*sy*sx + (y-1)*sx + x]; npred++; }
                if (z > 0) { pred += buf[(z-1)*sy*sx + y*sx + x]; npred++; }
                if (npred > 0) pred = (pred + npred/2) / npred;
                residuals[idx] = buf[idx] - (uint8_t)pred;
            }

    /* Step 2: histogram packing — find unique symbols */
    int freq[256] = {0};
    for (size_t i = 0; i < count; i++) freq[residuals[i]]++;
    int nsymbols = 0;
    uint8_t symbol_map[256];   /* value → packed index */
    uint8_t symbol_unmap[256]; /* packed index → value */
    memset(symbol_map, 0, 256);
    for (int i = 0; i < 256; i++) {
        if (freq[i] > 0) {
            symbol_map[i] = (uint8_t)nsymbols;
            symbol_unmap[nsymbols] = (uint8_t)i;
            nsymbols++;
        }
    }

    /* Pack residuals using the narrow alphabet */
    uint8_t *packed = malloc(count);
    for (size_t i = 0; i < count; i++)
        packed[i] = symbol_map[residuals[i]];

    /* Step 3: build output — [nsymbols(1)] [unmap table(nsymbols)] [packed data(count)]
     * If nsymbols < 128, this is smaller than raw. For nsymbols=1, it's just 2 bytes overhead. */
    size_t out_size = 1 + nsymbols + count;

    /* If histogram packing doesn't help, store raw residuals */
    if (nsymbols >= 200) {
        /* Not worth packing — store raw residuals with a marker */
        free(packed);
        c3d_compressed_t result;
        result.size = count;
        result.data = residuals;
        free(buf);
        return result;
    }

    uint8_t *out = malloc(out_size);
    if (!out) { free(buf); free(residuals); free(packed); return fail; }
    out[0] = (uint8_t)nsymbols;
    memcpy(out + 1, symbol_unmap, nsymbols);
    memcpy(out + 1 + nsymbols, packed, count);

    free(buf); free(residuals); free(packed);
    return (c3d_compressed_t){ .data = out, .size = out_size };
}

/* ── Rate-distortion split/leaf decision ── */

/* Try compressing a region as a leaf, return the size. Returns SIZE_MAX on failure. */
static size_t try_leaf_size(const uint8_t *vol, int vsx, int vsy,
                             int ox, int oy, int oz, int sx, int sy, int sz,
                             const c3d_octree_params_t *params) {
    c3d_compressed_t comp = compress_leaf_data(vol, vsx, vsy, ox, oy, oz, sx, sy, sz, params);
    size_t result = comp.data ? (5 + comp.size) : (size_t)-1; /* 1 type + 4 size + data */
    free(comp.data);
    return result;
}

/* Estimate the size of compressing a region as a branch (sum of children).
 * This is recursive but only goes one level deep for the estimate. */
static size_t estimate_branch_size(const uint8_t *vol, int vsx, int vsy, int vsz,
                                    int ox, int oy, int oz, int sx, int sy, int sz,
                                    const c3d_octree_params_t *params) {
    int hx = sx / 2, hy = sy / 2, hz = sz / 2;
    if (hx < params->min_leaf_dim) hx = sx;
    if (hy < params->min_leaf_dim) hy = sy;
    if (hz < params->min_leaf_dim) hz = sz;

    int nx = (hx < sx) ? 2 : 1;
    int ny = (hy < sy) ? 2 : 1;
    int nz = (hz < sz) ? 2 : 1;

    size_t total = 2; /* type + split_mask */
    for (int cz = 0; cz < nz; cz++)
        for (int cy = 0; cy < ny; cy++)
            for (int cx = 0; cx < nx; cx++) {
                int cox = ox + cx * hx;
                int coy = oy + cy * hy;
                int coz = oz + cz * hz;
                int csx = (cx == nx-1) ? (sx - cx*hx) : hx;
                int csy = (cy == ny-1) ? (sy - cy*hy) : hy;
                int csz = (cz == nz-1) ? (sz - cz*hz) : hz;

                /* Quick check: is child uniform/zero? */
                double mean, stddev;
                int all_zero;
                region_stats(vol, vsx, vsy, cox, coy, coz, csx, csy, csz, &mean, &stddev, &all_zero);

                if (all_zero)
                    total += 1; /* ZERO node */
                else if (stddev <= params->uniform_threshold)
                    total += 2; /* UNIFORM + value */
                else
                    total += try_leaf_size(vol, vsx, vsy, cox, coy, coz, csx, csy, csz, params);
            }

    return total;
}

/* ── Recursive octree compression ── */

static void octree_compress_rec(const uint8_t *vol, int vsx, int vsy, int vsz,
                                 int ox, int oy, int oz, int sx, int sy, int sz,
                                 const c3d_octree_params_t *params, obuf_t *out, int depth) {
    size_t count = (size_t)sx * sy * sz;
    double mean, stddev;
    int all_zero;
    region_stats(vol, vsx, vsy, ox, oy, oz, sx, sy, sz, &mean, &stddev, &all_zero);

    /* ZERO */
    if (all_zero) { obuf_u8(out, NODE_ZERO); return; }

    /* UNIFORM */
    if (stddev <= params->uniform_threshold) {
        obuf_u8(out, NODE_UNIFORM);
        obuf_u8(out, (uint8_t)(mean + 0.5));
        return;
    }

    /* Can we split? */
    int min_dim = params->min_leaf_dim;
    if (min_dim < 2) min_dim = 2;
    int can_split = (sx >= min_dim * 2) || (sy >= min_dim * 2) || (sz >= min_dim * 2);

    if (!can_split || depth >= 10) {
        /* Forced leaf */
        c3d_compressed_t comp = compress_leaf_data(vol, vsx, vsy, ox, oy, oz, sx, sy, sz, params);
        if (comp.data) {
            obuf_u8(out, NODE_LEAF);
            obuf_u16(out, (uint16_t)sx);
            obuf_u16(out, (uint16_t)sy);
            obuf_u16(out, (uint16_t)sz);
            obuf_u32(out, (uint32_t)comp.size);
            obuf_write(out, comp.data, comp.size);
            free(comp.data);
        }
        return;
    }

    /* Rate-distortion decision: leaf vs branch.
     * For large regions, always prefer splitting — the octree's ability to
     * zero-out/uniform empty sub-regions is the whole point.
     * Only consider leaf for regions that are small enough to compress well. */
    int force_split = 0;

    /* Always split large regions — leaf compression of 64³+ is expensive
     * and doesn't benefit from spatial adaptation */
    if (count > 32*32*32 && can_split) force_split = 1;

    /* For 32³ and smaller, do actual RD comparison */
    if (!force_split && count <= 32*32*32) {
        size_t leaf_cost = try_leaf_size(vol, vsx, vsy, ox, oy, oz, sx, sy, sz, params);
        leaf_cost += 11; /* type(1) + dims(6) + size(4) */

        size_t branch_cost = estimate_branch_size(vol, vsx, vsy, vsz, ox, oy, oz, sx, sy, sz, params);

        /* If leaf is cheaper (or close enough), use leaf */
        if (leaf_cost <= branch_cost * 11 / 10) {
            c3d_compressed_t comp = compress_leaf_data(vol, vsx, vsy, ox, oy, oz, sx, sy, sz, params);
            if (comp.data) {
                obuf_u8(out, NODE_LEAF);
                obuf_u16(out, (uint16_t)sx);
                obuf_u16(out, (uint16_t)sy);
                obuf_u16(out, (uint16_t)sz);
                obuf_u32(out, (uint32_t)comp.size);
                obuf_write(out, comp.data, comp.size);
                free(comp.data);
                return;
            }
        }
    }

    /* BRANCH: split */
    obuf_u8(out, NODE_BRANCH);

    int hx = (sx >= min_dim * 2) ? sx / 2 : sx;
    int hy = (sy >= min_dim * 2) ? sy / 2 : sy;
    int hz = (sz >= min_dim * 2) ? sz / 2 : sz;
    int nx = (hx < sx) ? 2 : 1;
    int ny = (hy < sy) ? 2 : 1;
    int nz = (hz < sz) ? 2 : 1;

    uint8_t split_mask = 0;
    if (nx > 1) split_mask |= 0x01;
    if (ny > 1) split_mask |= 0x02;
    if (nz > 1) split_mask |= 0x04;
    obuf_u8(out, split_mask);

    for (int cz = 0; cz < nz; cz++)
        for (int cy = 0; cy < ny; cy++)
            for (int cx = 0; cx < nx; cx++) {
                int cox = ox + cx * hx;
                int coy = oy + cy * hy;
                int coz = oz + cz * hz;
                int csx = (cx == nx-1) ? (sx - cx*hx) : hx;
                int csy = (cy == ny-1) ? (sy - cy*hy) : hy;
                int csz = (cz == nz-1) ? (sz - cz*hz) : hz;
                octree_compress_rec(vol, vsx, vsy, vsz, cox, coy, coz, csx, csy, csz, params, out, depth+1);
            }
}

/* ── Public API ── */

c3d_compressed_t c3d_octree_compress(const uint8_t *input,
    int sx, int sy, int sz, const c3d_octree_params_t *params) {
    c3d_compressed_t fail = { NULL, 0 };
    if (!input || !params || sx < 1 || sy < 1 || sz < 1) return fail;

    c3d_octree_params_t p = *params;
    if (p.uniform_threshold < 0) p.uniform_threshold = 2.0f;
    if (p.min_leaf_dim < 2) p.min_leaf_dim = 4;
    if (p.quality < 1) p.quality = 80;

    obuf_t out;
    obuf_init(&out);

    /* Header: magic(4) + dims(6) + quality(1) + threshold(1) + reserved(4) = 16 */
    uint8_t hdr[C3O_HEADER_SIZE] = {0};
    hdr[0] = C3O_MAGIC_0; hdr[1] = C3O_MAGIC_1;
    hdr[2] = C3O_MAGIC_2; hdr[3] = C3O_MAGIC_3;
    hdr[4] = (uint8_t)sx; hdr[5] = (uint8_t)(sx >> 8);
    hdr[6] = (uint8_t)sy; hdr[7] = (uint8_t)(sy >> 8);
    hdr[8] = (uint8_t)sz; hdr[9] = (uint8_t)(sz >> 8);
    hdr[10] = (uint8_t)p.quality;
    hdr[11] = (uint8_t)(int)(p.uniform_threshold);
    obuf_write(&out, hdr, C3O_HEADER_SIZE);

    octree_compress_rec(input, sx, sy, sz, 0, 0, 0, sx, sy, sz, &p, &out, 0);

    out.data = realloc(out.data, out.size);
    return (c3d_compressed_t){ .data = out.data, .size = out.size };
}

/* ── Recursive decompression ── */

static int octree_decompress_rec(ibuf_t *in, uint8_t *vol, int vsx, int vsy,
                                   int ox, int oy, int oz, int sx, int sy, int sz) {
    if (in->pos >= in->size) return -1;
    uint8_t type = ibuf_u8(in);

    switch (type) {
    case NODE_ZERO:
        for (int z = 0; z < sz; z++)
            for (int y = 0; y < sy; y++)
                memset(&vol[(oz+z)*vsy*vsx + (oy+y)*vsx + ox], 0, sx);
        return 0;

    case NODE_UNIFORM: {
        uint8_t val = ibuf_u8(in);
        for (int z = 0; z < sz; z++)
            for (int y = 0; y < sy; y++)
                memset(&vol[(oz+z)*vsy*vsx + (oy+y)*vsx + ox], val, sx);
        return 0;
    }

    case NODE_LEAF: {
        uint16_t lsx = ibuf_u16(in);
        uint16_t lsy = ibuf_u16(in);
        uint16_t lsz = ibuf_u16(in);
        uint32_t data_size = ibuf_u32(in);
        if (in->pos + data_size > in->size) return -1;
        (void)lsx; (void)lsy; (void)lsz; /* dims stored for validation */

        size_t count = (size_t)sx * sy * sz;
        const uint8_t *leaf_data = in->data + in->pos;
        in->pos += data_size;

        /* Tiled C3D blocks (for leaves >= 32³ that are multiples of 32) */
        if (data_size >= 5 && leaf_data[0] == 0xFE) {
            int bx = leaf_data[1], by = leaf_data[2], bz = leaf_data[3];
            int nblocks = leaf_data[4];
            if (nblocks != bx * by * bz) return -1;
            if (data_size < 5 + (size_t)nblocks * 4) return -1;

            const uint8_t *p = leaf_data + 5;
            uint32_t *sizes = malloc(nblocks * sizeof(uint32_t));
            for (int i = 0; i < nblocks; i++) {
                sizes[i] = p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
                p += 4;
            }

            for (int ibz = 0; ibz < bz; ibz++)
                for (int iby = 0; iby < by; iby++)
                    for (int ibx = 0; ibx < bx; ibx++) {
                        int bi = ibz*by*bx + iby*bx + ibx;
                        uint8_t block[32*32*32];
                        if (c3d_decompress(p, sizes[bi], block) == 0) {
                            for (int z = 0; z < 32; z++)
                                for (int y = 0; y < 32; y++)
                                    memcpy(&vol[(oz+ibz*32+z)*vsy*vsx + (oy+iby*32+y)*vsx + (ox+ibx*32)],
                                           &block[z*32*32 + y*32], 32);
                        }
                        p += sizes[bi];
                    }
            free(sizes);
            return 0;
        }

        /* Single 32³ C3D block */
        if (sx == 32 && sy == 32 && sz == 32) {
            uint8_t block[32*32*32];
            if (c3d_decompress(leaf_data, data_size, block) == 0) {
                scatter_region(block, vol, vsx, vsy, ox, oy, oz, 32, 32, 32);
                return 0;
            }
        }

        /* Check for histogram-packed format: first byte is nsymbols */
        if (data_size >= 2 && leaf_data[0] > 0 && leaf_data[0] < 200) {
            int nsymbols = leaf_data[0];
            size_t expected = 1 + nsymbols + count;
            if (data_size == expected) {
                const uint8_t *unmap = leaf_data + 1;
                const uint8_t *packed = leaf_data + 1 + nsymbols;

                /* Unpack + undo prediction */
                uint8_t *residuals = malloc(count);
                for (size_t i = 0; i < count; i++) {
                    uint8_t sym = packed[i];
                    residuals[i] = (sym < nsymbols) ? unmap[sym] : 0;
                }

                /* Undo 3D causal prediction */
                uint8_t *buf = malloc(count);
                for (int z = 0; z < sz; z++)
                    for (int y = 0; y < sy; y++)
                        for (int x = 0; x < sx; x++) {
                            int idx = z*sy*sx + y*sx + x;
                            int pred = 0, npred = 0;
                            if (x > 0) { pred += buf[z*sy*sx + y*sx + (x-1)]; npred++; }
                            if (y > 0) { pred += buf[z*sy*sx + (y-1)*sx + x]; npred++; }
                            if (z > 0) { pred += buf[(z-1)*sy*sx + y*sx + x]; npred++; }
                            if (npred > 0) pred = (pred + npred/2) / npred;
                            buf[idx] = (uint8_t)pred + residuals[idx];
                        }

                scatter_region(buf, vol, vsx, vsy, ox, oy, oz, sx, sy, sz);
                free(residuals); free(buf);
                return 0;
            }
        }

        /* Fallback: raw residuals (data_size == count) */
        if (data_size == count) {
            uint8_t *buf = malloc(count);
            /* Undo simple prediction */
            buf[0] = leaf_data[0];
            for (size_t i = 1; i < count; i++)
                buf[i] = buf[i-1] + leaf_data[i]; /* undo delta */
            scatter_region(buf, vol, vsx, vsy, ox, oy, oz, sx, sy, sz);
            free(buf);
            return 0;
        }

        /* Last resort: treat as raw data */
        {
            uint8_t *buf = malloc(count);
            size_t copy = (data_size < count) ? data_size : count;
            memcpy(buf, leaf_data, copy);
            if (copy < count) memset(buf + copy, 0, count - copy);
            scatter_region(buf, vol, vsx, vsy, ox, oy, oz, sx, sy, sz);
            free(buf);
        }
        return 0;
    }

    case NODE_BRANCH: {
        uint8_t split_mask = ibuf_u8(in);
        int hx = (split_mask & 1) ? sx / 2 : sx;
        int hy = (split_mask & 2) ? sy / 2 : sy;
        int hz = (split_mask & 4) ? sz / 2 : sz;
        int nx = (split_mask & 1) ? 2 : 1;
        int ny = (split_mask & 2) ? 2 : 1;
        int nz = (split_mask & 4) ? 2 : 1;

        for (int cz = 0; cz < nz; cz++)
            for (int cy = 0; cy < ny; cy++)
                for (int cx = 0; cx < nx; cx++) {
                    int rc = octree_decompress_rec(in, vol, vsx, vsy,
                        ox + cx*hx, oy + cy*hy, oz + cz*hz,
                        (cx == nx-1) ? (sx - cx*hx) : hx,
                        (cy == ny-1) ? (sy - cy*hy) : hy,
                        (cz == nz-1) ? (sz - cz*hz) : hz);
                    if (rc != 0) return rc;
                }
        return 0;
    }

    default: return -1;
    }
}

int c3d_octree_decompress(const uint8_t *compressed, size_t compressed_size, uint8_t *output) {
    if (!compressed || !output || compressed_size < C3O_HEADER_SIZE) return -1;
    if (compressed[0] != C3O_MAGIC_0 || compressed[1] != C3O_MAGIC_1 ||
        compressed[2] != C3O_MAGIC_2) return -1;
    /* Accept both v1 (0x01) and v2 (0x02) */
    if (compressed[3] != 0x01 && compressed[3] != 0x02) return -1;

    int sx = (int)compressed[4] | ((int)compressed[5] << 8);
    int sy = (int)compressed[6] | ((int)compressed[7] << 8);
    int sz = (int)compressed[8] | ((int)compressed[9] << 8);

    memset(output, 0, (size_t)sx * sy * sz);
    ibuf_t in = { compressed, compressed_size, C3O_HEADER_SIZE };
    return octree_decompress_rec(&in, output, sx, sy, 0, 0, 0, sx, sy, sz);
}

int c3d_octree_get_dims(const uint8_t *compressed, size_t compressed_size,
                         int *sx, int *sy, int *sz) {
    if (!compressed || compressed_size < C3O_HEADER_SIZE) return -1;
    if (compressed[0] != C3O_MAGIC_0 || compressed[1] != C3O_MAGIC_1 ||
        compressed[2] != C3O_MAGIC_2) return -1;
    if (sx) *sx = (int)compressed[4] | ((int)compressed[5] << 8);
    if (sy) *sy = (int)compressed[6] | ((int)compressed[7] << 8);
    if (sz) *sz = (int)compressed[8] | ((int)compressed[9] << 8);
    return 0;
}

/* ── Stats ── */

static void octree_stats_rec(ibuf_t *in, c3d_octree_stats_t *s, int depth) {
    if (in->pos >= in->size) return;
    uint8_t type = ibuf_u8(in);
    s->total_nodes++;
    if (depth > s->max_depth) s->max_depth = depth;

    switch (type) {
    case NODE_ZERO:
        s->uniform_nodes++; s->uniform_bytes += 1; break;
    case NODE_UNIFORM:
        ibuf_u8(in); s->uniform_nodes++; s->uniform_bytes += 2; break;
    case NODE_LEAF: {
        ibuf_u16(in); ibuf_u16(in); ibuf_u16(in); /* dims */
        uint32_t dsz = ibuf_u32(in);
        in->pos += dsz;
        s->leaf_nodes++; s->leaf_bytes += 11 + dsz;
        break;
    }
    case NODE_BRANCH: {
        uint8_t mask = ibuf_u8(in);
        s->branch_nodes++; s->tree_bytes += 2;
        int n = ((mask&1)?2:1) * ((mask&2)?2:1) * ((mask&4)?2:1);
        for (int i = 0; i < n; i++) octree_stats_rec(in, s, depth+1);
        break;
    }
    }
}

int c3d_octree_stats(const uint8_t *compressed, size_t compressed_size,
                      c3d_octree_stats_t *stats) {
    if (!compressed || !stats || compressed_size < C3O_HEADER_SIZE) return -1;
    memset(stats, 0, sizeof(*stats));
    ibuf_t in = { compressed, compressed_size, C3O_HEADER_SIZE };
    octree_stats_rec(&in, stats, 0);
    return 0;
}
