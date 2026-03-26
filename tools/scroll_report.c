/* scroll_report.c — Load blosc-compressed zarr chunks, compress with C3D
 * at various quality levels, and generate quality reports.
 *
 * Usage: scroll_report <chunk1.raw> [chunk2.raw ...]
 */

#include "compress3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <blosc.h>

#define CHUNK_DIM 128
#define CHUNK_SIZE (CHUNK_DIM * CHUNK_DIM * CHUNK_DIM) /* 2,097,152 bytes */
#define BLOCK_DIM 32
#define BLOCK_SIZE (BLOCK_DIM * BLOCK_DIM * BLOCK_DIM)

static uint8_t *load_blosc_chunk(const char *path, size_t *raw_compressed_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t *compressed = (uint8_t *)malloc(fsize);
    fread(compressed, 1, fsize, fp);
    fclose(fp);
    *raw_compressed_size = (size_t)fsize;

    /* Decompress with blosc */
    uint8_t *output = (uint8_t *)malloc(CHUNK_SIZE);
    if (!output) { free(compressed); return NULL; }

    int nbytes = blosc_decompress(compressed, output, CHUNK_SIZE);
    free(compressed);

    if (nbytes < 0) {
        fprintf(stderr, "Blosc decompress failed for %s (error %d)\n", path, nbytes);
        free(output);
        return NULL;
    }

    return output;
}

/* Extract a 32^3 block from a 128^3 chunk */
static void extract_block(const uint8_t *chunk, int bx, int by, int bz, uint8_t *block) {
    for (int z = 0; z < BLOCK_DIM; z++)
        for (int y = 0; y < BLOCK_DIM; y++)
            for (int x = 0; x < BLOCK_DIM; x++)
                block[z * BLOCK_DIM * BLOCK_DIM + y * BLOCK_DIM + x] =
                    chunk[(bz * BLOCK_DIM + z) * CHUNK_DIM * CHUNK_DIM
                        + (by * BLOCK_DIM + y) * CHUNK_DIM
                        + (bx * BLOCK_DIM + x)];
}

static void analyze_block(const uint8_t *block, const char *label) {
    /* Basic stats */
    int vmin = 255, vmax = 0;
    double sum = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (block[i] < vmin) vmin = block[i];
        if (block[i] > vmax) vmax = block[i];
        sum += block[i];
    }
    double mean = sum / BLOCK_SIZE;
    double var = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        double d = block[i] - mean;
        var += d * d;
    }
    var /= BLOCK_SIZE;

    printf("  %s: min=%d max=%d mean=%.1f stddev=%.1f\n",
           label, vmin, vmax, mean, sqrt(var));
}

static void run_quality_sweep(const uint8_t *block, const char *label) {
    printf("\n── %s ──\n", label);
    analyze_block(block, "input");

    int qualities[] = {20, 40, 60, 80, 90, 95, 100, 101};
    int nq = sizeof(qualities) / sizeof(qualities[0]);

    printf("  %-6s %8s %8s %8s %8s %6s %8s %8s %8s\n",
           "Q", "Size", "Ratio", "BPV", "PSNR", "MaxE", "SSIM", "Corr", "HistInt");
    printf("  %-6s %8s %8s %8s %8s %6s %8s %8s %8s\n",
           "---", "----", "-----", "---", "----", "----", "----", "----", "-------");

    for (int qi = 0; qi < nq; qi++) {
        int q = qualities[qi];
        c3d_compressed_t comp = c3d_compress(block, q);
        if (!comp.data) { printf("  q=%-3d  FAILED\n", q); continue; }

        uint8_t recon[BLOCK_SIZE];
        c3d_decompress(comp.data, comp.size, recon);

        c3d_quality_report_t r;
        c3d_quality_report(block, recon, BLOCK_SIZE, comp.size, &r);

        printf("  q=%-3d %7zu %7.1f:1 %6.3f %7.2f %5d %8.6f %8.6f %8.4f\n",
               q, comp.size, r.compression_ratio, r.bits_per_voxel,
               r.psnr, r.max_error, r.ssim, r.correlation, r.histogram_intersection);

        free(comp.data);
    }

    /* Also try wavelet and auto modes */
    printf("\n  Transform comparison at q=80:\n");
    printf("  %-10s %8s %8s %8s %8s\n", "Mode", "Size", "PSNR", "SSIM", "Ratio");

    /* DCT */
    {
        c3d_compressed_t comp = c3d_compress(block, 80);
        uint8_t recon[BLOCK_SIZE];
        c3d_decompress(comp.data, comp.size, recon);
        printf("  %-10s %7zu %7.2f %8.6f %7.1f:1\n", "DCT",
               comp.size, c3d_psnr(block, recon, BLOCK_SIZE),
               c3d_ssim(block, recon), (double)BLOCK_SIZE / comp.size);
        free(comp.data);
    }

    /* Wavelet */
    {
        c3d_compressed_t comp = c3d_compress_wavelet(block, 80);
        uint8_t recon[BLOCK_SIZE];
        c3d_decompress(comp.data, comp.size, recon);
        printf("  %-10s %7zu %7.2f %8.6f %7.1f:1\n", "Wavelet",
               comp.size, c3d_psnr(block, recon, BLOCK_SIZE),
               c3d_ssim(block, recon), (double)BLOCK_SIZE / comp.size);
        free(comp.data);
    }

    /* Auto */
    {
        c3d_compressed_t comp = c3d_compress_auto(block, 80);
        uint8_t recon[BLOCK_SIZE];
        c3d_decompress(comp.data, comp.size, recon);
        printf("  %-10s %7zu %7.2f %8.6f %7.1f:1\n", "Auto",
               comp.size, c3d_psnr(block, recon, BLOCK_SIZE),
               c3d_ssim(block, recon), (double)BLOCK_SIZE / comp.size);
        free(comp.data);
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
        if (!chunk) continue;

        printf("\n════════════════════════════════════════════════════════════════\n");
        printf("File: %s\n", argv[f]);
        printf("Blosc compressed: %zu bytes (%.1f:1 vs raw 128^3)\n",
               blosc_size, (double)CHUNK_SIZE / blosc_size);

        /* Test a few blocks from different positions */
        const int positions[][3] = {{0,0,0}, {2,2,2}, {1,1,1}, {3,3,3}};
        int npos = 4;

        for (int p = 0; p < npos; p++) {
            uint8_t block[BLOCK_SIZE];
            extract_block(chunk, positions[p][0], positions[p][1], positions[p][2], block);

            char label[64];
            snprintf(label, sizeof(label), "block(%d,%d,%d) from %s",
                     positions[p][0], positions[p][1], positions[p][2], argv[f]);
            run_quality_sweep(block, label);
        }

        /* Multiscale test on a 64^3 sub-volume */
        printf("\n── Multiscale test (64^3 sub-volume) ──\n");
        uint8_t *sub64 = (uint8_t *)malloc(64*64*64);
        for (int z = 0; z < 64; z++)
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++)
                    sub64[z*64*64 + y*64 + x] = chunk[z*CHUNK_DIM*CHUNK_DIM + y*CHUNK_DIM + x];

        int test_qualities[] = {80, 90, 101};
        for (int qi = 0; qi < 3; qi++) {
            int q = test_qualities[qi];
            c3d_multiscale_params_t params = {
                .quality = q,
                .upsample_method = C3D_UPSAMPLE_TRILINEAR,
                .num_threads = 0
            };
            c3d_compressed_t comp = c3d_multiscale_compress(sub64, 64, 64, 64, &params);
            if (!comp.data) { printf("  q=%d multiscale FAILED\n", q); continue; }

            uint8_t *recon = (uint8_t *)malloc(64*64*64);
            c3d_multiscale_decompress(comp.data, comp.size, recon);

            size_t vol = 64*64*64;
            printf("  Multiscale q=%-3d: size=%zu (%.1f:1) PSNR=%.2f dB SSIM=%.6f\n",
                   q, comp.size, (double)vol / comp.size,
                   c3d_psnr(sub64, recon, vol),
                   c3d_ssim_volume(sub64, recon, 64, 64, 64));

            free(recon);
            free(comp.data);
        }
        free(sub64);

        free(chunk);
    }

    blosc_destroy();
    return 0;
}
