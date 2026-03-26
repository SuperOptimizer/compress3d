#include "compress3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <blosc.h>

#define CHUNK_DIM 128
#define CHUNK_SIZE (CHUNK_DIM * CHUNK_DIM * CHUNK_DIM)

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <chunk.raw>\n", argv[0]); return 1; }
    blosc_init();

    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { fprintf(stderr, "Can't open %s\n", argv[1]); return 1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *comp = malloc(fsize);
    if (fread(comp, 1, fsize, fp) != (size_t)fsize) { fclose(fp); return 1; }
    fclose(fp);

    uint8_t *chunk = malloc(CHUNK_SIZE);
    if (blosc_decompress(comp, chunk, CHUNK_SIZE) < 0) { fprintf(stderr, "Blosc failed\n"); return 1; }
    free(comp);

    printf("File: %s (blosc: %ld bytes, %.1f:1)\n\n", argv[1], fsize, (double)CHUNK_SIZE / fsize);

    /* Test octree at various quality/threshold settings */
    printf("═══ OCTREE COMPRESSION OF FULL 128³ CHUNK ═══\n");
    printf("%-5s %-6s %8s %8s %7s %6s %8s %8s  %6s %6s %6s %5s\n",
           "Q", "Thresh", "Size", "Ratio", "PSNR", "MAE", "Enc(us)", "Dec(us)",
           "Unif", "Leaf", "Branch", "Depth");

    int qualities[] = {101, 95, 80, 60, 40, 20};
    float thresholds[] = {0.0, 1.0, 2.0, 5.0, 10.0};

    for (int qi = 0; qi < 6; qi++) {
        for (int ti = 0; ti < 5; ti++) {
            c3d_octree_params_t params = {
                .quality = qualities[qi],
                .uniform_threshold = thresholds[ti],
                .min_leaf_dim = 4,
                .step = 0
            };

            double t0 = now_us();
            c3d_compressed_t oc = c3d_octree_compress(chunk, CHUNK_DIM, CHUNK_DIM, CHUNK_DIM, &params);
            double t1 = now_us();
            if (!oc.data) { printf("q=%-3d t=%.0f FAILED\n", qualities[qi], thresholds[ti]); continue; }

            uint8_t *output = malloc(CHUNK_SIZE);
            double t2 = now_us();
            int rc = c3d_octree_decompress(oc.data, oc.size, output);
            double t3 = now_us();

            double psnr = (rc == 0) ? c3d_psnr(chunk, output, CHUNK_SIZE) : 0;
            double mae = (rc == 0) ? c3d_mae(chunk, output, CHUNK_SIZE) : 0;

            c3d_octree_stats_t stats;
            c3d_octree_stats(oc.data, oc.size, &stats);

            printf("q=%-3d t=%-4.0f %7zu %7.1f:1 %6.2f %5.2f %7.0f %7.0f  %5d %5d %5d %4d\n",
                   qualities[qi], thresholds[ti],
                   oc.size, (double)CHUNK_SIZE / oc.size,
                   psnr, mae, t1-t0, t3-t2,
                   stats.uniform_nodes, stats.leaf_nodes, stats.branch_nodes, stats.max_depth);

            free(output); free(oc.data);
        }
    }

    /* Compare vs fixed-block approach */
    printf("\n═══ COMPARISON: FIXED 32³ BLOCKS vs OCTREE ═══\n");
    int nblocks = (CHUNK_DIM/32) * (CHUNK_DIM/32) * (CHUNK_DIM/32); /* 64 blocks */

    for (int qi = 0; qi < 4; qi++) {
        int q = qualities[qi];

        /* Fixed blocks */
        size_t fixed_total = 0;
        double t0 = now_us();
        for (int bz = 0; bz < 4; bz++)
            for (int by = 0; by < 4; by++)
                for (int bx = 0; bx < 4; bx++) {
                    uint8_t block[32*32*32];
                    for (int z = 0; z < 32; z++)
                        for (int y = 0; y < 32; y++)
                            memcpy(&block[z*32*32+y*32],
                                   &chunk[(bz*32+z)*128*128+(by*32+y)*128+bx*32], 32);
                    c3d_compressed_t c = c3d_compress(block, q);
                    if (c.data) { fixed_total += c.size; free(c.data); }
                }
        double t1 = now_us();

        /* Octree */
        c3d_octree_params_t params = { .quality = q, .uniform_threshold = 2.0, .min_leaf_dim = 4, .step = 0 };
        double t2 = now_us();
        c3d_compressed_t oc = c3d_octree_compress(chunk, CHUNK_DIM, CHUNK_DIM, CHUNK_DIM, &params);
        double t3 = now_us();

        printf("q=%-3d  Fixed: %7zu (%5.1f:1, %.0fms)  Octree: %7zu (%5.1f:1, %.0fms)  Savings: %.0f%%\n",
               q, fixed_total, (double)CHUNK_SIZE/fixed_total, (t1-t0)/1000,
               oc.data ? oc.size : 0, oc.data ? (double)CHUNK_SIZE/oc.size : 0, (t3-t2)/1000,
               oc.data ? 100.0 * (1.0 - (double)oc.size / fixed_total) : 0);
        free(oc.data);
    }

    blosc_destroy();
    free(chunk);
    return 0;
}
