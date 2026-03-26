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

static void test_lossless_roundtrip(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t output[C3D_BLOCK_VOXELS];

    /* Fill with a gradient pattern */
    for (int i = 0; i < C3D_BLOCK_VOXELS; i++)
        input[i] = (uint8_t)(i % 256);

    c3d_compressed_t comp = c3d_compress(input, 101);
    ASSERT(comp.data != NULL, "lossless compress returned non-NULL");
    ASSERT(comp.size > 0, "lossless compress returned non-zero size");

    int rc = c3d_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "lossless decompress succeeded");
    ASSERT(memcmp(input, output, C3D_BLOCK_VOXELS) == 0, "lossless roundtrip is exact");

    free(comp.data);
}

static void test_lossy_quality_range(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t output[C3D_BLOCK_VOXELS];

    for (int i = 0; i < C3D_BLOCK_VOXELS; i++)
        input[i] = (uint8_t)(rand() % 256);

    /* Low quality should produce smaller output than high quality */
    c3d_compressed_t low = c3d_compress(input, 10);
    c3d_compressed_t high = c3d_compress(input, 90);
    ASSERT(low.data != NULL && high.data != NULL, "lossy compress succeeded");
    ASSERT(low.size < high.size, "lower quality produces smaller output");

    /* Both should decompress without error */
    ASSERT(c3d_decompress(low.data, low.size, output) == 0, "low quality decompress ok");
    ASSERT(c3d_decompress(high.data, high.size, output) == 0, "high quality decompress ok");

    free(low.data);
    free(high.data);
}

static void test_compress_to(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t compressed[65536];
    uint8_t output[C3D_BLOCK_VOXELS];

    for (int i = 0; i < C3D_BLOCK_VOXELS; i++)
        input[i] = (uint8_t)(i * 7 % 256);

    size_t sz = c3d_compress_to(input, 101, compressed, sizeof(compressed));
    ASSERT(sz > 0, "compress_to returned non-zero");

    int rc = c3d_decompress_to(compressed, sz, output);
    ASSERT(rc == 0, "decompress_to succeeded");
    ASSERT(memcmp(input, output, C3D_BLOCK_VOXELS) == 0, "compress_to lossless roundtrip exact");
}

static void test_get_size(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    memset(input, 128, C3D_BLOCK_VOXELS);

    c3d_compressed_t comp = c3d_compress(input, 50);
    ASSERT(comp.data != NULL, "compress for get_size ok");

    int sz = c3d_get_size(comp.data, comp.size);
    ASSERT(sz == C3D_BLOCK_SIZE, "get_size returns 32");

    free(comp.data);
}

static void test_compress_bound(void) {
    size_t bound = c3d_compress_bound();
    ASSERT(bound > 0, "compress_bound is positive");
    ASSERT(bound >= C3D_BLOCK_VOXELS, "compress_bound >= block voxels");
}

static void test_ssim(void) {
    uint8_t a[C3D_BLOCK_VOXELS];
    uint8_t b[C3D_BLOCK_VOXELS];

    for (int i = 0; i < C3D_BLOCK_VOXELS; i++) {
        a[i] = (uint8_t)(i % 256);
        b[i] = a[i];
    }
    double ssim = c3d_ssim(a, b);
    ASSERT(fabs(ssim - 1.0) < 1e-6, "SSIM of identical blocks is 1.0");

    /* Perturb b slightly */
    for (int i = 0; i < C3D_BLOCK_VOXELS; i++)
        b[i] = (uint8_t)((a[i] + 1) % 256);
    ssim = c3d_ssim(a, b);
    ASSERT(ssim > 0.9 && ssim < 1.0, "SSIM of similar blocks is high but < 1");
}

static void test_wavelet_roundtrip(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t output[C3D_BLOCK_VOXELS];

    for (int i = 0; i < C3D_BLOCK_VOXELS; i++)
        input[i] = (uint8_t)(i % 256);

    c3d_compressed_t comp = c3d_compress_wavelet(input, 101);
    ASSERT(comp.data != NULL, "wavelet lossless compress ok");

    int rc = c3d_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "wavelet decompress ok");
    ASSERT(memcmp(input, output, C3D_BLOCK_VOXELS) == 0, "wavelet lossless roundtrip exact");

    free(comp.data);
}

static void test_workspace(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t compressed[65536];
    uint8_t output[C3D_BLOCK_VOXELS];

    for (int i = 0; i < C3D_BLOCK_VOXELS; i++)
        input[i] = (uint8_t)(i % 256);

    c3d_workspace_t *ws = c3d_workspace_create();
    ASSERT(ws != NULL, "workspace create ok");

    size_t sz = c3d_compress_ws(input, 101, compressed, sizeof(compressed), ws);
    ASSERT(sz > 0, "compress_ws returned non-zero");

    int rc = c3d_decompress_ws(compressed, sz, output, ws);
    ASSERT(rc == 0, "decompress_ws ok");
    ASSERT(memcmp(input, output, C3D_BLOCK_VOXELS) == 0, "workspace lossless roundtrip exact");

    c3d_workspace_free(ws);
}

static void test_null_safety(void) {
    uint8_t buf[C3D_BLOCK_VOXELS];
    c3d_compressed_t comp = c3d_compress(NULL, 50);
    ASSERT(comp.data == NULL && comp.size == 0, "compress(NULL) returns {NULL,0}");

    ASSERT(c3d_decompress(NULL, 0, buf) == -1, "decompress(NULL) returns -1");
    ASSERT(c3d_get_size(NULL, 0) == -1, "get_size(NULL) returns -1");
    ASSERT(c3d_ssim(NULL, buf) == 0.0, "ssim(NULL,...) returns 0.0");
}

int main(void) {
    test_lossless_roundtrip();
    test_lossy_quality_range();
    test_compress_to();
    test_get_size();
    test_compress_bound();
    test_ssim();
    test_wavelet_roundtrip();
    test_workspace();
    test_null_safety();

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
