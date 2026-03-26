/* ratio_sweep.c — Comprehensive compression sweep testing step API and
 * native multiscale decode across 1x through 32768x (1x1x1) range.
 *
 * Usage: ratio_sweep <chunk1.raw> [chunk2.raw ...]
 */

#include "compress3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <blosc.h>

#define CHUNK_DIM 128
#define CHUNK_SIZE (CHUNK_DIM * CHUNK_DIM * CHUNK_DIM)
#define BLOCK_DIM 32
#define BLOCK_SIZE (BLOCK_DIM * BLOCK_DIM * BLOCK_DIM)

static uint8_t *load_blosc_chunk(const char *path, size_t *raw_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *comp = malloc(fsize);
    if (fread(comp, 1, fsize, fp) != (size_t)fsize) { free(comp); fclose(fp); return NULL; }
    fclose(fp);
    *raw_size = fsize;
    uint8_t *out = malloc(CHUNK_SIZE);
    if (blosc_decompress(comp, out, CHUNK_SIZE) < 0) { free(comp); free(out); return NULL; }
    free(comp);
    return out;
}

static void extract_block(const uint8_t *chunk, int bx, int by, int bz, uint8_t *block) {
    for (int z = 0; z < BLOCK_DIM; z++)
        for (int y = 0; y < BLOCK_DIM; y++)
            for (int x = 0; x < BLOCK_DIM; x++)
                block[z*BLOCK_DIM*BLOCK_DIM + y*BLOCK_DIM + x] =
                    chunk[(bz*BLOCK_DIM+z)*CHUNK_DIM*CHUNK_DIM + (by*BLOCK_DIM+y)*CHUNK_DIM + (bx*BLOCK_DIM+x)];
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* Reference downsample for comparison: simple 2x2x2 average in spatial domain */
static void downsample_ref(const uint8_t *in, int dim, uint8_t *out) {
    int half = dim / 2;
    for (int z = 0; z < half; z++)
        for (int y = 0; y < half; y++)
            for (int x = 0; x < half; x++) {
                int sum = 0;
                for (int dz = 0; dz < 2; dz++)
                    for (int dy = 0; dy < 2; dy++)
                        for (int dx = 0; dx < 2; dx++)
                            sum += in[(z*2+dz)*dim*dim + (y*2+dy)*dim + (x*2+dx)];
                out[z*half*half + y*half + x] = (uint8_t)((sum + 4) / 8);
            }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <chunk1.raw> [chunk2.raw ...]\n", argv[0]);
        return 1;
    }
    blosc_init();

    for (int f = 1; f < argc; f++) {
        size_t blosc_size;
        uint8_t *chunk = load_blosc_chunk(argv[f], &blosc_size);
        if (!chunk) { fprintf(stderr, "Failed: %s\n", argv[f]); continue; }

        printf("\n════════════════════════════════════════════════════════════════════\n");
        printf("File: %s\n\n", argv[f]);

        uint8_t block[BLOCK_SIZE];
        extract_block(chunk, 1, 1, 1, block);

        /* Get lossless baseline */
        c3d_compressed_t lossless = c3d_compress(block, 101);
        size_t lossless_size = lossless.data ? lossless.size : BLOCK_SIZE;
        printf("Lossless: %zu bytes (%.1f:1 vs raw %d)\n\n",
               lossless_size, (double)BLOCK_SIZE / lossless_size, BLOCK_SIZE);

        /* ── Part 1: Step-based compression sweep ── */
        printf("═══ STEP-BASED COMPRESSION (c3d_compress_step) ═══\n");
        printf("%-8s %8s %8s %7s %7s %6s %5s %8s %8s %8s\n",
               "Step", "Size", "Ratio", "BPV", "PSNR", "MAE", "MaxE", "SSIM", "Enc(us)", "Dec(us)");

        float steps[] = {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 75.0, 100.0,
                          150.0, 200.0, 300.0, 400.0, 500.0};
        int nsteps = sizeof(steps) / sizeof(steps[0]);

        for (int s = 0; s < nsteps; s++) {
            double t0 = now_us();
            c3d_compressed_t comp = c3d_compress_step(block, steps[s], 0);
            double t1 = now_us();
            if (!comp.data) { printf("%-8.1f  FAILED\n", steps[s]); continue; }

            uint8_t recon[BLOCK_SIZE];
            double t2 = now_us();
            c3d_decompress(comp.data, comp.size, recon);
            double t3 = now_us();

            printf("%-8.1f %7zu %7.1f:1 %6.3f %6.2f %5.2f %5d %8.6f %7.0f %7.0f\n",
                   steps[s], comp.size, (double)BLOCK_SIZE / comp.size,
                   (double)(comp.size * 8) / BLOCK_SIZE,
                   c3d_psnr(block, recon, BLOCK_SIZE),
                   c3d_mae(block, recon, BLOCK_SIZE),
                   c3d_max_error(block, recon, BLOCK_SIZE),
                   c3d_ssim(block, recon),
                   t1-t0, t3-t2);
            free(comp.data);
        }

        /* ── Part 2: Coefficient-truncated compression ── */
        printf("\n═══ COEFFICIENT TRUNCATION (c3d_compress_step with max_coeffs) ═══\n");
        printf("%-10s %8s %8s %7s %7s %6s %5s %8s\n",
               "MaxCoeff", "Size", "Ratio", "BPV", "PSNR", "MAE", "MaxE", "SSIM");

        int coeff_budgets[] = {1, 8, 27, 64, 125, 216, 512, 1000, 2000, 4096, 8000, 16384, 32768};
        int nbudgets = sizeof(coeff_budgets) / sizeof(coeff_budgets[0]);

        for (int b = 0; b < nbudgets; b++) {
            c3d_compressed_t comp = c3d_compress_step(block, 2.0, coeff_budgets[b]);
            if (!comp.data) continue;

            uint8_t recon[BLOCK_SIZE];
            c3d_decompress(comp.data, comp.size, recon);

            int dim_equiv = 1;
            while (dim_equiv * dim_equiv * dim_equiv < coeff_budgets[b] && dim_equiv < 32)
                dim_equiv *= 2;

            printf("%-5d(%2d³) %6zu %7.1f:1 %6.3f %6.2f %5.2f %5d %8.6f\n",
                   coeff_budgets[b], dim_equiv, comp.size,
                   (double)BLOCK_SIZE / comp.size,
                   (double)(comp.size * 8) / BLOCK_SIZE,
                   c3d_psnr(block, recon, BLOCK_SIZE),
                   c3d_mae(block, recon, BLOCK_SIZE),
                   c3d_max_error(block, recon, BLOCK_SIZE),
                   c3d_ssim(block, recon));
            free(comp.data);
        }

        /* ── Part 3: Native multiscale decode (c3d_decode_at_resolution) ── */
        printf("\n═══ NATIVE MULTISCALE DECODE (DCT sub-cube) ═══\n");
        printf("%-8s %8s %8s %8s\n", "OutDim", "Voxels", "PSNR_vs_ref", "SSIM_vs_ref");

        /* Compress at moderate quality */
        c3d_compressed_t comp80 = c3d_compress(block, 80);
        if (comp80.data) {
            int dims[] = {1, 2, 4, 8, 16, 32};
            int ndims = 6;

            /* Build reference downscales from full decode */
            uint8_t full[BLOCK_SIZE];
            c3d_decompress(comp80.data, comp80.size, full);

            for (int d = 0; d < ndims; d++) {
                int M = dims[d];
                int M3 = M * M * M;
                uint8_t *partial = (uint8_t *)malloc(M3);

                int rc = c3d_decode_at_resolution(comp80.data, comp80.size, M, partial);

                if (rc == 0 && M >= 2) {
                    /* Build reference downsample from full decode */
                    uint8_t *ref = (uint8_t *)malloc(M3);
                    uint8_t *tmp = (uint8_t *)malloc(BLOCK_SIZE);
                    memcpy(tmp, full, BLOCK_SIZE);
                    int cur = 32;
                    while (cur > M) {
                        int next = cur / 2;
                        uint8_t *ds = (uint8_t *)malloc(next*next*next);
                        downsample_ref(tmp, cur, ds);
                        free(tmp);
                        tmp = ds;
                        cur = next;
                    }
                    memcpy(ref, tmp, M3);
                    free(tmp);

                    printf("%-3d³     %7d %10.2f %12.6f\n", M, M3,
                           c3d_psnr(ref, partial, M3),
                           M3 >= 512 ? c3d_ssim_volume(ref, partial, M, M, M) : -1.0);
                    free(ref);
                } else if (rc == 0 && M == 1) {
                    printf("%-3d³     %7d    (DC=%d, ref_mean=%.0f)\n",
                           M, M3, partial[0],
                           c3d_mse(block, block, 0) /* placeholder */);
                    /* Just report the DC value */
                    double sum = 0;
                    for (int i = 0; i < BLOCK_SIZE; i++) sum += full[i];
                    printf("         (full_mean=%.1f, partial_dc=%d)\n", sum/BLOCK_SIZE, partial[0]);
                } else {
                    printf("%-3d³     FAILED (rc=%d)\n", M, rc);
                }
                free(partial);
            }
            free(comp80.data);
        }

        /* ── Part 4: Target ratio sweep using c3d_compress_ratio ── */
        printf("\n═══ TARGET RATIO SWEEP (c3d_compress_ratio — combined step+truncation) ═══\n");
        printf("%-8s %8s %8s %7s %7s %6s %5s %8s\n",
               "Target", "Size", "Actual", "BPV", "PSNR", "MAE", "MaxE", "SSIM");

        float ratios[] = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0};
        int nratios = sizeof(ratios) / sizeof(ratios[0]);

        for (int r = 0; r < nratios; r++) {
            if (ratios[r] <= 1.0f) {
                printf("%-8s %7zu %7.1f:1 %6.3f %6s %5s %5s %8s\n",
                       "1x", lossless_size, (double)BLOCK_SIZE/lossless_size,
                       (double)(lossless_size*8)/BLOCK_SIZE, "inf", "0.00", "0", "1.000000");
                continue;
            }

            c3d_compressed_t comp = c3d_compress_ratio(block, ratios[r]);
            if (!comp.data) {
                char label[16]; snprintf(label, sizeof(label), "%.0fx", ratios[r]);
                printf("%-8s  FAILED\n", label);
                continue;
            }

            uint8_t recon[BLOCK_SIZE];
            c3d_decompress(comp.data, comp.size, recon);

            char label[16]; snprintf(label, sizeof(label), "%.0fx", ratios[r]);
            double actual_ratio = (double)lossless_size / comp.size;
            printf("%-8s %7zu %7.1f:1 %6.3f %6.2f %5.2f %5d %8.6f\n",
                   label, comp.size, actual_ratio,
                   (double)(comp.size*8)/BLOCK_SIZE,
                   c3d_psnr(block, recon, BLOCK_SIZE),
                   c3d_mae(block, recon, BLOCK_SIZE),
                   c3d_max_error(block, recon, BLOCK_SIZE),
                   c3d_ssim(block, recon));
            free(comp.data);
        }

        free(lossless.data);
        free(chunk);
    }

    blosc_destroy();
    return 0;
}
