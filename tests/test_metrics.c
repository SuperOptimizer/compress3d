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

static void test_mse_identical(void) {
    uint8_t a[1024], b[1024];
    for (int i = 0; i < 1024; i++) a[i] = b[i] = (uint8_t)(i % 256);
    ASSERT(c3d_mse(a, b, 1024) == 0.0, "MSE of identical buffers is 0");
    ASSERT(c3d_psnr(a, b, 1024) == INFINITY, "PSNR of identical buffers is INFINITY");
    ASSERT(c3d_mae(a, b, 1024) == 0.0, "MAE of identical is 0");
    ASSERT(c3d_max_error(a, b, 1024) == 0, "max_error of identical is 0");
    ASSERT(c3d_rmse(a, b, 1024) == 0.0, "RMSE of identical is 0");
}

static void test_mse_known(void) {
    uint8_t a[4] = {0, 0, 0, 0};
    uint8_t b[4] = {1, 1, 1, 1};
    ASSERT(c3d_mse(a, b, 4) == 1.0, "MSE of all-off-by-1 is 1.0");

    uint8_t c[4] = {10, 10, 10, 10};
    double mse = c3d_mse(a, c, 4);
    ASSERT(mse == 100.0, "MSE of all-off-by-10 is 100.0");
}

static void test_psnr_range(void) {
    uint8_t a[C3D_BLOCK_VOXELS], b[C3D_BLOCK_VOXELS];
    for (int i = 0; i < C3D_BLOCK_VOXELS; i++) {
        a[i] = (uint8_t)(i % 200 + 20); /* stay in [20, 219] to avoid wrap */
        b[i] = a[i] + 1;
    }
    double psnr = c3d_psnr(a, b, C3D_BLOCK_VOXELS);
    /* MSE=1 → PSNR = 10*log10(65025) ≈ 48.13 dB */
    ASSERT(psnr > 48.0 && psnr < 49.0, "PSNR of off-by-1 is ~48.1 dB");
}

static void test_mae(void) {
    uint8_t a[4] = {0, 10, 20, 30};
    uint8_t b[4] = {5, 5, 25, 25};
    double mae = c3d_mae(a, b, 4);
    ASSERT(mae == 5.0, "MAE is 5.0");
}

static void test_max_error(void) {
    uint8_t a[4] = {0, 100, 200, 255};
    uint8_t b[4] = {0, 100, 100, 255};
    ASSERT(c3d_max_error(a, b, 4) == 100, "max_error is 100");
}

static void test_snr(void) {
    uint8_t a[1024], b[1024];
    for (int i = 0; i < 1024; i++) {
        a[i] = (uint8_t)(128 + 50.0 * sin((double)i * 0.01));
        b[i] = a[i]; /* identical first */
    }
    double snr = c3d_snr(a, b, 1024);
    ASSERT(snr == INFINITY, "SNR of identical is INFINITY");

    /* Add small noise */
    for (int i = 0; i < 1024; i++) {
        int v = (int)a[i] + ((i % 3) - 1); /* -1, 0, 1 */
        b[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    snr = c3d_snr(a, b, 1024);
    ASSERT(snr > 20.0, "SNR with small noise > 20 dB");
}

static void test_correlation(void) {
    uint8_t a[256], b[256];
    for (int i = 0; i < 256; i++) {
        a[i] = (uint8_t)i;
        b[i] = (uint8_t)i;
    }
    double r = c3d_correlation(a, b, 256);
    ASSERT(r > 0.999, "correlation of identical is ~1.0");

    /* Anticorrelated */
    for (int i = 0; i < 256; i++) b[i] = (uint8_t)(255 - i);
    r = c3d_correlation(a, b, 256);
    ASSERT(r < -0.999, "correlation of inverted is ~-1.0");
}

static void test_nrmse(void) {
    uint8_t a[100], b[100];
    for (int i = 0; i < 100; i++) {
        a[i] = (uint8_t)(i + 50);
        b[i] = a[i];
    }
    ASSERT(c3d_nrmse(a, b, 100) == 0.0, "NRMSE of identical is 0");

    for (int i = 0; i < 100; i++) b[i] = (uint8_t)(a[i] + 1);
    double nrmse = c3d_nrmse(a, b, 100);
    ASSERT(nrmse > 0.0 && nrmse < 0.1, "NRMSE of off-by-1 is small");
}

static void test_histogram_intersection(void) {
    uint8_t a[256], b[256];
    for (int i = 0; i < 256; i++) a[i] = b[i] = (uint8_t)i;
    double hi = c3d_histogram_intersection(a, b, 256);
    ASSERT(hi > 0.99, "histogram intersection of identical is ~1.0");

    /* Completely different distributions */
    for (int i = 0; i < 256; i++) { a[i] = 0; b[i] = 255; }
    hi = c3d_histogram_intersection(a, b, 256);
    ASSERT(hi < 0.01, "histogram intersection of disjoint is ~0.0");
}

static void test_compression_ratio(void) {
    ASSERT(c3d_compression_ratio(1000, 100) == 10.0, "ratio 1000/100 = 10");
    ASSERT(c3d_compression_ratio(1000, 0) == 0.0, "ratio div-by-zero safe");
    ASSERT(c3d_bits_per_voxel(100, 1000) == 0.8, "bpv 100 bytes / 1000 voxels = 0.8");
}

static void test_quality_report(void) {
    uint8_t orig[C3D_BLOCK_VOXELS], recon[C3D_BLOCK_VOXELS];
    for (int i = 0; i < C3D_BLOCK_VOXELS; i++) {
        orig[i] = (uint8_t)(128 + 50.0 * sin((double)i * 0.01));
    }

    /* Compress and decompress */
    c3d_compressed_t comp = c3d_compress(orig, 80);
    ASSERT(comp.data != NULL, "compress for report");
    c3d_decompress(comp.data, comp.size, recon);

    c3d_quality_report_t report;
    int rc = c3d_quality_report(orig, recon, C3D_BLOCK_VOXELS, comp.size, &report);
    ASSERT(rc == 0, "quality report ok");

    printf("\n  Quality report for q=80 compress/decompress:\n");
    c3d_quality_report_print(&report);

    ASSERT(report.psnr > 30.0, "report PSNR > 30 dB");
    ASSERT(report.ssim > 0.9, "report SSIM > 0.9");
    ASSERT(report.correlation > 0.99, "report correlation > 0.99");
    ASSERT(report.histogram_intersection > 0.8, "report hist > 0.8");
    ASSERT(report.compression_ratio > 1.0, "report compressed smaller");
    ASSERT(report.bits_per_voxel > 0.0, "report bpv > 0");

    free(comp.data);
}

static void test_ssim_volume(void) {
    /* 16x16x16 volume */
    int dim = 16;
    size_t sz = (size_t)dim * dim * dim;
    uint8_t *a = (uint8_t *)malloc(sz);
    uint8_t *b = (uint8_t *)malloc(sz);
    for (size_t i = 0; i < sz; i++) a[i] = b[i] = (uint8_t)(i % 256);

    double ssim = c3d_ssim_volume(a, b, dim, dim, dim);
    ASSERT(ssim > 0.999, "SSIM volume of identical ~1.0");

    /* Perturb b */
    for (size_t i = 0; i < sz; i++) b[i] = (uint8_t)((a[i] + 2) % 256);
    ssim = c3d_ssim_volume(a, b, dim, dim, dim);
    ASSERT(ssim > 0.8 && ssim < 1.0, "SSIM volume of similar is high");

    free(a); free(b);
}

static void test_ms_ssim(void) {
    int dim = 32;
    size_t sz = (size_t)dim * dim * dim;
    uint8_t *a = (uint8_t *)malloc(sz);
    uint8_t *b = (uint8_t *)malloc(sz);
    for (size_t i = 0; i < sz; i++) a[i] = b[i] = (uint8_t)(i % 256);

    double ms = c3d_ms_ssim(a, b, dim, dim, dim);
    ASSERT(ms > 0.99, "MS-SSIM of identical ~1.0");

    for (size_t i = 0; i < sz; i++) b[i] = (uint8_t)((a[i] + 3) % 256);
    ms = c3d_ms_ssim(a, b, dim, dim, dim);
    ASSERT(ms > 0.5 && ms < 1.0, "MS-SSIM of similar is moderate-high");

    free(a); free(b);
}

static void test_null_safety(void) {
    uint8_t buf[64];
    ASSERT(c3d_mse(NULL, buf, 64) == 0.0, "mse null safe");
    ASSERT(c3d_psnr(NULL, buf, 64) == INFINITY, "psnr null safe");
    ASSERT(c3d_mae(NULL, buf, 64) == 0.0, "mae null safe");
    ASSERT(c3d_max_error(NULL, buf, 64) == 0, "max_error null safe");
    ASSERT(c3d_snr(NULL, buf, 64) == 0.0, "snr null safe");
    ASSERT(c3d_correlation(NULL, buf, 64) == 0.0, "correlation null safe");
    ASSERT(c3d_histogram_intersection(NULL, buf, 64) == 0.0, "hist null safe");

    c3d_quality_report_t r;
    ASSERT(c3d_quality_report(NULL, buf, 64, 0, &r) == -1, "report null safe");
}

int main(void) {
    test_mse_identical();
    test_mse_known();
    test_psnr_range();
    test_mae();
    test_max_error();
    test_snr();
    test_correlation();
    test_nrmse();
    test_histogram_intersection();
    test_compression_ratio();
    test_quality_report();
    test_ssim_volume();
    test_ms_ssim();
    test_null_safety();

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
