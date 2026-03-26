/* analyze_data.c — Deep analysis of scroll CT data characteristics to guide
 * compression improvements. Analyzes histogram, spatial correlation, sparsity,
 * gradient distributions, and inter-block redundancy.
 */

#include "compress3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <blosc.h>

#define CHUNK_DIM 128
#define CHUNK_SIZE (CHUNK_DIM * CHUNK_DIM * CHUNK_DIM)
#define BD 32
#define BS (BD * BD * BD)

static uint8_t *load_blosc_chunk(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *comp = malloc(fsize);
    if (fread(comp, 1, fsize, fp) != (size_t)fsize) { free(comp); fclose(fp); return NULL; }
    fclose(fp);
    uint8_t *out = malloc(CHUNK_SIZE);
    if (blosc_decompress(comp, out, CHUNK_SIZE) < 0) { free(comp); free(out); return NULL; }
    free(comp);
    return out;
}

static void extract_block(const uint8_t *chunk, int bx, int by, int bz, uint8_t *block) {
    for (int z = 0; z < BD; z++)
        for (int y = 0; y < BD; y++)
            for (int x = 0; x < BD; x++)
                block[z*BD*BD + y*BD + x] =
                    chunk[(bz*BD+z)*CHUNK_DIM*CHUNK_DIM + (by*BD+y)*CHUNK_DIM + (bx*BD+x)];
}

static void analyze_block(const uint8_t *block, const char *label) {
    printf("\n── %s ──\n", label);

    /* Histogram */
    int hist[256] = {0};
    double sum = 0, sum2 = 0;
    int vmin = 255, vmax = 0;
    for (int i = 0; i < BS; i++) {
        hist[block[i]]++;
        sum += block[i];
        sum2 += (double)block[i] * block[i];
        if (block[i] < vmin) vmin = block[i];
        if (block[i] > vmax) vmax = block[i];
    }
    double mean = sum / BS;
    double var = sum2 / BS - mean * mean;

    printf("  Stats: min=%d max=%d mean=%.1f stddev=%.1f\n", vmin, vmax, mean, sqrt(var));

    /* Sparsity: fraction of voxels below various thresholds */
    int below[9] = {0};
    int thresholds[] = {1, 5, 10, 20, 30, 50, 80, 100, 128};
    for (int i = 0; i < BS; i++)
        for (int t = 0; t < 9; t++)
            if (block[i] < thresholds[t]) below[t]++;
    printf("  Sparsity: ");
    for (int t = 0; t < 9; t++)
        printf("<%d:%.0f%% ", thresholds[t], 100.0 * below[t] / BS);
    printf("\n");

    /* Entropy */
    double entropy = 0;
    for (int i = 0; i < 256; i++) {
        if (hist[i] > 0) {
            double p = (double)hist[i] / BS;
            entropy -= p * log2(p);
        }
    }
    printf("  Entropy: %.3f bits/voxel (theoretical min: %.0f bytes)\n",
           entropy, entropy * BS / 8.0);

    /* Spatial correlation: avg abs difference to 6 neighbors */
    double grad_sum = 0;
    int grad_count = 0;
    for (int z = 0; z < BD; z++)
        for (int y = 0; y < BD; y++)
            for (int x = 0; x < BD; x++) {
                int v = block[z*BD*BD + y*BD + x];
                if (x > 0) { grad_sum += abs(v - block[z*BD*BD + y*BD + (x-1)]); grad_count++; }
                if (y > 0) { grad_sum += abs(v - block[z*BD*BD + (y-1)*BD + x]); grad_count++; }
                if (z > 0) { grad_sum += abs(v - block[(z-1)*BD*BD + y*BD + x]); grad_count++; }
            }
    printf("  Avg neighbor diff: %.2f\n", grad_sum / grad_count);

    /* Prediction residual entropy (LOCO-I style) */
    int res_hist[256] = {0};
    for (int z = 0; z < BD; z++)
        for (int y = 0; y < BD; y++)
            for (int x = 0; x < BD; x++) {
                int v = block[z*BD*BD + y*BD + x];
                int pred;
                if (x > 0 && y > 0 && z > 0) {
                    int a = block[z*BD*BD + y*BD + (x-1)];
                    int b = block[z*BD*BD + (y-1)*BD + x];
                    int d = block[(z-1)*BD*BD + y*BD + x];
                    pred = (a + b + d) / 3;
                } else if (x > 0) pred = block[z*BD*BD + y*BD + (x-1)];
                else if (y > 0) pred = block[z*BD*BD + (y-1)*BD + x];
                else if (z > 0) pred = block[(z-1)*BD*BD + y*BD + x];
                else pred = 0;
                res_hist[(uint8_t)(v - pred)]++;
            }
    double res_entropy = 0;
    for (int i = 0; i < 256; i++) {
        if (res_hist[i] > 0) {
            double p = (double)res_hist[i] / BS;
            res_entropy -= p * log2(p);
        }
    }
    printf("  Prediction residual entropy: %.3f bits/voxel (%.0f bytes)\n",
           res_entropy, res_entropy * BS / 8.0);

    /* DCT energy distribution: what % of energy is in each frequency shell */
    float vol[BS];
    for (int i = 0; i < BS; i++) vol[i] = (float)block[i] - 128.0f;

    /* Forward DCT (manually, since dct3d_forward_all is internal) */
    /* We'll use compress + inspect sizes instead */
    printf("\n  Compression at various levels:\n");
    printf("  %-12s %8s %7s %6s\n", "Method", "Size", "Ratio", "BPV");

    /* Lossless */
    c3d_compressed_t cl = c3d_compress(block, 101);
    if (cl.data) {
        printf("  %-12s %7zu %6.1f:1 %5.3f\n", "Lossless", cl.size, (double)BS/cl.size, (double)(cl.size*8)/BS);
        free(cl.data);
    }

    /* Various quality levels */
    int qs[] = {95, 90, 80, 60, 40, 20, 1};
    for (int qi = 0; qi < 7; qi++) {
        c3d_compressed_t c = c3d_compress(block, qs[qi]);
        if (c.data) {
            char label2[16]; snprintf(label2, sizeof(label2), "q=%d", qs[qi]);
            printf("  %-12s %7zu %6.1f:1 %5.3f\n", label2, c.size, (double)BS/c.size, (double)(c.size*8)/BS);
            free(c.data);
        }
    }

    /* Step-based with coefficient truncation */
    printf("\n  Coefficient truncation (step=0.5, near-lossless quant):\n");
    printf("  %-10s %8s %7s %6s %6s\n", "Coeffs", "Size", "Ratio", "BPV", "PSNR");
    int budgets[] = {1, 8, 64, 512, 4096, 32768};
    for (int bi = 0; bi < 6; bi++) {
        c3d_compressed_t c = c3d_compress_step(block, 0.5, budgets[bi]);
        if (c.data) {
            uint8_t recon[BS];
            c3d_decompress(c.data, c.size, recon);
            printf("  %-5d(%2d³) %7zu %6.1f:1 %5.3f %5.1f\n",
                   budgets[bi], (int)round(cbrt(budgets[bi])),
                   c.size, (double)BS/c.size, (double)(c.size*8)/BS,
                   c3d_psnr(block, recon, BS));
            free(c.data);
        }
    }

    /* Ideal: entropy of prediction residuals = theoretical lossless floor */
    printf("\n  Theoretical floors:\n");
    printf("    Raw entropy:        %.3f bpv → %.0f bytes\n", entropy, entropy * BS / 8.0);
    printf("    Residual entropy:   %.3f bpv → %.0f bytes\n", res_entropy, res_entropy * BS / 8.0);
    printf("    Current lossless:   %.3f bpv\n", cl.data ? (double)(cl.size*8)/BS : 0);
    printf("    Gap (current - theoretical): %.3f bpv\n",
           cl.data ? (double)(cl.size*8)/BS - res_entropy : 0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <chunk1.raw> [chunk2.raw ...]\n", argv[0]);
        return 1;
    }
    blosc_init();

    for (int f = 1; f < argc; f++) {
        uint8_t *chunk = load_blosc_chunk(argv[f]);
        if (!chunk) continue;

        printf("════════════════════════════════════════════════════════════════\n");
        printf("File: %s\n", argv[f]);

        int positions[][3] = {{0,0,0}, {1,1,1}, {2,2,2}, {3,3,3}};
        for (int p = 0; p < 4; p++) {
            uint8_t block[BS];
            extract_block(chunk, positions[p][0], positions[p][1], positions[p][2], block);
            char label[64];
            snprintf(label, sizeof(label), "block(%d,%d,%d)", positions[p][0], positions[p][1], positions[p][2]);
            analyze_block(block, label);
        }
        free(chunk);
    }

    blosc_destroy();
    return 0;
}
