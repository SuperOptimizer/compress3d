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

static void test_downsample_upsample(void) {
    /* 4x4x4 volume */
    uint8_t input[64];
    for (int i = 0; i < 64; i++) input[i] = (uint8_t)(i * 4);

    uint8_t down[8]; /* 2x2x2 */
    int rc = c3d_downsample_2x(input, 4, 4, 4, down);
    ASSERT(rc == 0, "downsample_2x succeeds");

    /* Each output voxel is average of 8 input voxels */
    ASSERT(down[0] > 0, "downsampled value is nonzero");

    uint8_t up[64]; /* back to 4x4x4 */
    rc = c3d_upsample_2x(down, 2, 2, 2, C3D_UPSAMPLE_TRILINEAR, up);
    ASSERT(rc == 0, "upsample_2x succeeds");

    /* Check that upsampled is a reasonable approximation */
    double mse = 0;
    for (int i = 0; i < 64; i++) {
        double d = (double)input[i] - (double)up[i];
        mse += d * d;
    }
    mse /= 64;
    ASSERT(mse < 2000, "upsample is a reasonable approximation");
}

static void test_downsample_null(void) {
    uint8_t buf[8];
    ASSERT(c3d_downsample_2x(NULL, 2, 2, 2, buf) == -1, "downsample null input");
    ASSERT(c3d_upsample_2x(NULL, 2, 2, 2, 0, buf) == -1, "upsample null input");
}

static void test_multiscale_small_lossless(void) {
    /* 32x32x32 volume = one C3D block, produces 6 levels (1,2,4,8,16,32) */
    int dim = 32;
    size_t vol_size = (size_t)dim * dim * dim;
    uint8_t *input = (uint8_t *)malloc(vol_size);
    ASSERT(input != NULL, "alloc input");

    /* Fill with a gradient */
    for (size_t i = 0; i < vol_size; i++)
        input[i] = (uint8_t)(i % 256);

    c3d_multiscale_params_t params = {
        .quality = 101,  /* lossless */
        .upsample_method = C3D_UPSAMPLE_TRILINEAR,
        .num_threads = 0
    };

    c3d_compressed_t comp = c3d_multiscale_compress(input, dim, dim, dim, &params);
    ASSERT(comp.data != NULL, "multiscale compress succeeded");
    ASSERT(comp.size > 0, "multiscale compress non-zero size");

    /* Read header */
    c3d_multiscale_header_t hdr;
    int rc = c3d_multiscale_header(comp.data, comp.size, &hdr);
    ASSERT(rc == 0, "header parse ok");
    ASSERT(hdr.num_levels == 6, "32^3 volume has 6 levels (1,2,4,8,16,32)");
    ASSERT(hdr.native_x == 32, "native_x == 32");
    ASSERT(hdr.levels[0].dim_x == 1, "level 0 is 1x1x1");
    ASSERT(hdr.levels[5].dim_x == 32, "level 5 is 32x32x32");

    /* Decompress native level */
    uint8_t *output = (uint8_t *)malloc(vol_size);
    ASSERT(output != NULL, "alloc output");

    rc = c3d_multiscale_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "multiscale decompress succeeded");

    /* With lossless residuals + lossy upsample, we expect close but not exact */
    int max_diff = 0;
    int diff_count = 0;
    for (size_t i = 0; i < vol_size; i++) {
        int d = abs((int)input[i] - (int)output[i]);
        if (d > max_diff) max_diff = d;
        if (d > 0) diff_count++;
    }
    printf("  [multiscale lossless] max_diff=%d diff_count=%d/%zu\n",
           max_diff, diff_count, vol_size);
    /* The residual encoding should keep error very low */
    ASSERT(max_diff <= 2, "max diff <= 2 (residual quantization + upsample)");

    free(input); free(output); free(comp.data);
}

static void test_multiscale_lossy(void) {
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
    ASSERT(comp.data != NULL, "lossy multiscale compress ok");

    uint8_t *output = (uint8_t *)malloc(vol_size);
    int rc = c3d_multiscale_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "lossy multiscale decompress ok");

    /* Compute PSNR */
    double mse = 0;
    for (size_t i = 0; i < vol_size; i++) {
        double d = (double)input[i] - (double)output[i];
        mse += d * d;
    }
    mse /= (double)vol_size;
    double psnr = (mse > 0) ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
    printf("  [multiscale lossy q=80] PSNR=%.1f dB\n", psnr);
    ASSERT(psnr > 25.0, "lossy PSNR > 25 dB");

    free(input); free(output); free(comp.data);
}

static void test_multiscale_progressive_decode(void) {
    int dim = 32;
    size_t vol_size = (size_t)dim * dim * dim;
    uint8_t *input = (uint8_t *)malloc(vol_size);

    for (size_t i = 0; i < vol_size; i++)
        input[i] = (uint8_t)(i % 256);

    c3d_multiscale_params_t params = {
        .quality = 90,
        .upsample_method = C3D_UPSAMPLE_TRILINEAR,
        .num_threads = 0
    };

    c3d_compressed_t comp = c3d_multiscale_compress(input, dim, dim, dim, &params);
    ASSERT(comp.data != NULL, "progressive compress ok");

    c3d_multiscale_header_t hdr;
    c3d_multiscale_header(comp.data, comp.size, &hdr);

    /* Decode each level progressively */
    c3d_level_cache_t *cache = c3d_level_cache_create();
    ASSERT(cache != NULL, "cache create ok");

    for (int lev = 0; lev < hdr.num_levels; lev++) {
        size_t lsz = (size_t)hdr.levels[lev].dim_x * hdr.levels[lev].dim_y * hdr.levels[lev].dim_z;
        uint8_t *buf = (uint8_t *)malloc(lsz);
        int rc = c3d_multiscale_decompress_level(comp.data, comp.size, lev, buf, cache);
        ASSERT(rc == 0, "progressive decode level ok");
        free(buf);
    }

    c3d_level_cache_free(cache);
    free(input); free(comp.data);
}

static void test_multiscale_64(void) {
    /* 64x64x64 — multiple C3D blocks per level */
    int dim = 64;
    size_t vol_size = (size_t)dim * dim * dim;
    uint8_t *input = (uint8_t *)malloc(vol_size);

    for (size_t i = 0; i < vol_size; i++)
        input[i] = (uint8_t)(128 + 30.0 * sin((double)i * 0.005));

    c3d_multiscale_params_t params = {
        .quality = 85,
        .upsample_method = C3D_UPSAMPLE_TRILINEAR,
        .num_threads = 0
    };

    c3d_compressed_t comp = c3d_multiscale_compress(input, dim, dim, dim, &params);
    ASSERT(comp.data != NULL, "64^3 multiscale compress ok");

    c3d_multiscale_header_t hdr;
    int rc = c3d_multiscale_header(comp.data, comp.size, &hdr);
    ASSERT(rc == 0, "64^3 header ok");
    ASSERT(hdr.num_levels == 7, "64^3 has 7 levels");
    ASSERT(hdr.levels[6].dim_x == 64, "top level is 64");

    uint8_t *output = (uint8_t *)malloc(vol_size);
    rc = c3d_multiscale_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "64^3 decompress ok");

    double mse = 0;
    for (size_t i = 0; i < vol_size; i++) {
        double d = (double)input[i] - (double)output[i];
        mse += d * d;
    }
    mse /= (double)vol_size;
    double psnr = (mse > 0) ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
    printf("  [multiscale 64^3 q=85] PSNR=%.1f dB, compressed=%zu bytes (%.1f:1)\n",
           psnr, comp.size, (double)vol_size / comp.size);
    ASSERT(psnr > 25.0, "64^3 PSNR > 25 dB");

    free(input); free(output); free(comp.data);
}

int main(void) {
    test_downsample_upsample();
    test_downsample_null();
    test_multiscale_small_lossless();
    test_multiscale_lossy();
    test_multiscale_progressive_decode();
    test_multiscale_64();

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
