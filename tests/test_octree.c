#include "compress3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
    else { passes++; } \
} while(0)

static int passes = 0;
static int failures = 0;

static void test_octree_uniform(void) {
    /* 64³ volume of all-128 — should compress to almost nothing */
    int dim = 64;
    size_t sz = (size_t)dim * dim * dim;
    uint8_t *input = (uint8_t *)malloc(sz);
    memset(input, 128, sz);

    c3d_octree_params_t params = { .quality = 80, .uniform_threshold = 2.0, .min_leaf_dim = 4, .step = 0 };
    c3d_compressed_t comp = c3d_octree_compress(input, dim, dim, dim, &params);
    ASSERT(comp.data != NULL, "octree compress uniform");
    ASSERT(comp.size < 100, "uniform 64^3 compresses to < 100 bytes");
    printf("  uniform 64^3: %zu bytes (%.0f:1)\n", comp.size, (double)sz / comp.size);

    uint8_t *output = (uint8_t *)malloc(sz);
    int rc = c3d_octree_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "octree decompress uniform");
    ASSERT(memcmp(input, output, sz) == 0, "uniform roundtrip exact");

    free(input); free(output); free(comp.data);
}

static void test_octree_zeros(void) {
    int dim = 128;
    size_t sz = (size_t)dim * dim * dim;
    uint8_t *input = (uint8_t *)calloc(sz, 1);

    c3d_octree_params_t params = { .quality = 101, .uniform_threshold = 2.0, .min_leaf_dim = 4, .step = 0 };
    c3d_compressed_t comp = c3d_octree_compress(input, dim, dim, dim, &params);
    ASSERT(comp.data != NULL, "octree compress zeros");
    ASSERT(comp.size < 50, "zero 128^3 compresses to < 50 bytes");
    printf("  zero 128^3: %zu bytes (%.0f:1)\n", comp.size, (double)sz / comp.size);

    uint8_t *output = (uint8_t *)malloc(sz);
    int rc = c3d_octree_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "octree decompress zeros");
    ASSERT(memcmp(input, output, sz) == 0, "zero roundtrip exact");

    free(input); free(output); free(comp.data);
}

static void test_octree_sparse(void) {
    /* 128³ volume: mostly zero with a few bright spots */
    int dim = 128;
    size_t sz = (size_t)dim * dim * dim;
    uint8_t *input = (uint8_t *)calloc(sz, 1);

    /* Place a bright sphere in the center */
    int cx = dim/2, cy = dim/2, cz = dim/2, r = 20;
    for (int z = 0; z < dim; z++)
        for (int y = 0; y < dim; y++)
            for (int x = 0; x < dim; x++) {
                int dx = x-cx, dy = y-cy, dz = z-cz;
                if (dx*dx + dy*dy + dz*dz < r*r)
                    input[z*dim*dim + y*dim + x] = 200;
            }

    c3d_octree_params_t params = { .quality = 90, .uniform_threshold = 2.0, .min_leaf_dim = 4, .step = 0 };
    c3d_compressed_t comp = c3d_octree_compress(input, dim, dim, dim, &params);
    ASSERT(comp.data != NULL, "octree compress sparse");

    /* The empty regions should compress away, so ratio should be very high */
    double ratio = (double)sz / comp.size;
    printf("  sparse 128^3 (sphere r=20): %zu bytes (%.0f:1)\n", comp.size, ratio);
    ASSERT(ratio > 10.0, "sparse data compresses > 10:1");

    uint8_t *output = (uint8_t *)malloc(sz);
    int rc = c3d_octree_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "octree decompress sparse");

    /* Check quality */
    double psnr = c3d_psnr(input, output, sz);
    printf("  sparse PSNR: %.2f dB\n", psnr);
    ASSERT(psnr > 30.0, "sparse PSNR > 30 dB");

    /* Check stats */
    c3d_octree_stats_t stats;
    c3d_octree_stats(comp.data, comp.size, &stats);
    printf("  stats: total=%d uniform=%d leaf=%d branch=%d depth=%d\n",
           stats.total_nodes, stats.uniform_nodes, stats.leaf_nodes,
           stats.branch_nodes, stats.max_depth);
    printf("  bytes: uniform=%zu leaf=%zu tree=%zu\n",
           stats.uniform_bytes, stats.leaf_bytes, stats.tree_bytes);
    ASSERT(stats.uniform_nodes > stats.leaf_nodes, "more uniform than leaf nodes (sparse data)");

    free(input); free(output); free(comp.data);
}

static void test_octree_gradient(void) {
    /* 64³ volume with a gradient — tests lossy compression quality */
    int dim = 64;
    size_t sz = (size_t)dim * dim * dim;
    uint8_t *input = (uint8_t *)malloc(sz);
    for (int z = 0; z < dim; z++)
        for (int y = 0; y < dim; y++)
            for (int x = 0; x < dim; x++)
                input[z*dim*dim + y*dim + x] = (uint8_t)((x + y + z) * 255 / (3 * (dim-1)));

    c3d_octree_params_t params = { .quality = 80, .uniform_threshold = 2.0, .min_leaf_dim = 4, .step = 0 };
    c3d_compressed_t comp = c3d_octree_compress(input, dim, dim, dim, &params);
    ASSERT(comp.data != NULL, "octree compress gradient");
    printf("  gradient 64^3: %zu bytes (%.1f:1)\n", comp.size, (double)sz / comp.size);

    uint8_t *output = (uint8_t *)malloc(sz);
    int rc = c3d_octree_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "octree decompress gradient");

    double psnr = c3d_psnr(input, output, sz);
    printf("  gradient PSNR: %.2f dB\n", psnr);
    ASSERT(psnr > 18.0, "gradient PSNR > 18 dB");

    free(input); free(output); free(comp.data);
}

static void test_octree_get_dims(void) {
    int dim = 100; /* non-power-of-2 */
    size_t sz = (size_t)dim * dim * dim;
    uint8_t *input = (uint8_t *)calloc(sz, 1);

    c3d_octree_params_t params = { .quality = 80, .uniform_threshold = 5.0, .min_leaf_dim = 4, .step = 0 };
    c3d_compressed_t comp = c3d_octree_compress(input, dim, dim, dim, &params);
    ASSERT(comp.data != NULL, "octree compress non-pow2");

    int sx, sy, sz2;
    int rc = c3d_octree_get_dims(comp.data, comp.size, &sx, &sy, &sz2);
    ASSERT(rc == 0, "get_dims ok");
    ASSERT(sx == dim && sy == dim && sz2 == dim, "dims match");

    free(input); free(comp.data);
}

static void test_octree_lossless(void) {
    /* Small 32³ volume, lossless */
    int dim = 32;
    size_t sz = (size_t)dim * dim * dim;
    uint8_t *input = (uint8_t *)malloc(sz);
    for (size_t i = 0; i < sz; i++) input[i] = (uint8_t)(i % 256);

    c3d_octree_params_t params = { .quality = 101, .uniform_threshold = 0.0, .min_leaf_dim = 4, .step = 0 };
    c3d_compressed_t comp = c3d_octree_compress(input, dim, dim, dim, &params);
    ASSERT(comp.data != NULL, "octree lossless compress");

    uint8_t *output = (uint8_t *)malloc(sz);
    int rc = c3d_octree_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "octree lossless decompress");
    ASSERT(memcmp(input, output, sz) == 0, "octree lossless exact roundtrip");

    printf("  lossless 32^3: %zu bytes (%.1f:1)\n", comp.size, (double)sz / comp.size);

    free(input); free(output); free(comp.data);
}

static void test_octree_null_safety(void) {
    c3d_octree_params_t params = { .quality = 80, .uniform_threshold = 2.0, .min_leaf_dim = 4, .step = 0 };
    c3d_compressed_t comp = c3d_octree_compress(NULL, 32, 32, 32, &params);
    ASSERT(comp.data == NULL, "null input");

    ASSERT(c3d_octree_decompress(NULL, 0, NULL) == -1, "null decompress");
    ASSERT(c3d_octree_get_dims(NULL, 0, NULL, NULL, NULL) == -1, "null get_dims");
}

int main(void) {
    test_octree_uniform();
    test_octree_zeros();
    test_octree_sparse();
    test_octree_gradient();
    test_octree_get_dims();
    test_octree_lossless();
    test_octree_null_safety();

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
