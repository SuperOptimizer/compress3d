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

/* Deterministic test volume */
static void fill_gradient(uint8_t *buf) {
    for (int i = 0; i < C3D_BLOCK_VOXELS; i++)
        buf[i] = (uint8_t)((i * 7 + 13) % 256);
}

static void test_progressive_roundtrip(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t output[C3D_BLOCK_VOXELS];
    fill_gradient(input);

    /* Progressive format is lossy-only (quality 1-100), 101 is rejected */
    c3d_compressed_t comp = c3d_compress_progressive(input, 100);
    ASSERT(comp.data != NULL, "progressive compress returned non-NULL");
    ASSERT(comp.size > 0, "progressive compress returned non-zero size");

    int rc = c3d_decompress_progressive(comp.data, comp.size, output);
    ASSERT(rc == 0, "progressive decompress succeeded");

    double psnr = c3d_psnr(input, output, C3D_BLOCK_VOXELS);
    ASSERT(psnr > 40.0, "progressive high-quality roundtrip has high PSNR");

    free(comp.data);
}

static void test_progressive_truncation(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t output[C3D_BLOCK_VOXELS];
    fill_gradient(input);

    c3d_compressed_t comp = c3d_compress_progressive(input, 80);
    ASSERT(comp.data != NULL, "progressive compress for truncation ok");

    /* Decode only first 50% of bytes -- should still produce valid output */
    size_t half = comp.size / 2;
    memset(output, 0, C3D_BLOCK_VOXELS);
    int rc = c3d_decompress_progressive(comp.data, half, output);
    ASSERT(rc == 0, "progressive truncated decompress succeeded");

    /* Truncated should still produce something reasonable (not all zeros) */
    int nonzero = 0;
    for (int i = 0; i < C3D_BLOCK_VOXELS; i++) {
        if (output[i] != 0) nonzero++;
    }
    ASSERT(nonzero > C3D_BLOCK_VOXELS / 4, "truncated progressive has substantial non-zero output");

    free(comp.data);
}

static void test_compress_meta(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    fill_gradient(input);

    c3d_metadata_t meta_in;
    memset(&meta_in, 0, sizeof(meta_in));
    meta_in.voxel_size[0] = 0.5f;
    meta_in.voxel_size[1] = 0.5f;
    meta_in.voxel_size[2] = 1.0f;
    meta_in.modality = 1; /* CT */
    meta_in.bits_per_voxel = 8;

    c3d_compressed_t comp = c3d_compress_meta(input, 80, &meta_in);
    ASSERT(comp.data != NULL, "compress_meta returned non-NULL");
    ASSERT(comp.size > 0, "compress_meta returned non-zero size");

    c3d_metadata_t meta_out;
    memset(&meta_out, 0, sizeof(meta_out));
    int rc = c3d_get_metadata(comp.data, comp.size, &meta_out);
    ASSERT(rc == 0, "get_metadata succeeded");
    ASSERT(fabsf(meta_out.voxel_size[0] - 0.5f) < 1e-6f, "voxel_size[0] roundtrip ok");
    ASSERT(fabsf(meta_out.voxel_size[1] - 0.5f) < 1e-6f, "voxel_size[1] roundtrip ok");
    ASSERT(fabsf(meta_out.voxel_size[2] - 1.0f) < 1e-6f, "voxel_size[2] roundtrip ok");
    ASSERT(meta_out.modality == 1, "modality roundtrip ok (CT)");

    /* Verify decompression still works */
    uint8_t output[C3D_BLOCK_VOXELS];
    rc = c3d_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "decompress after compress_meta ok");

    free(comp.data);
}

static void test_compress_target_psnr(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t output[C3D_BLOCK_VOXELS];
    fill_gradient(input);

    double target = 35.0;
    c3d_compressed_t comp = c3d_compress_target_psnr(input, target);
    ASSERT(comp.data != NULL, "target_psnr compress returned non-NULL");

    int rc = c3d_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "target_psnr decompress ok");

    double actual_psnr = c3d_psnr(input, output, C3D_BLOCK_VOXELS);
    ASSERT(actual_psnr >= target - 2.0, "actual PSNR >= target - 2dB");

    free(comp.data);
}

static void test_compress_target_size(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    fill_gradient(input);

    size_t target = 8000; /* target compressed size */
    c3d_compressed_t comp = c3d_compress_target_size(input, target);
    ASSERT(comp.data != NULL, "target_size compress returned non-NULL");
    ASSERT(comp.size <= target, "compressed size <= target size");

    free(comp.data);
}

static void test_compress_target_ssim(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t output[C3D_BLOCK_VOXELS];
    fill_gradient(input);

    double target = 0.90;
    c3d_compressed_t comp = c3d_compress_target_ssim(input, target);
    ASSERT(comp.data != NULL, "target_ssim compress returned non-NULL");

    int rc = c3d_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "target_ssim decompress ok");

    double actual_ssim = c3d_ssim(input, output);
    ASSERT(actual_ssim >= target - 0.02, "actual SSIM >= target - 0.02");

    free(comp.data);
}

static void test_compress_auto(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t output[C3D_BLOCK_VOXELS];
    fill_gradient(input);

    c3d_compressed_t comp = c3d_compress_auto(input, 80);
    ASSERT(comp.data != NULL, "compress_auto returned non-NULL");
    ASSERT(comp.size > 0, "compress_auto returned non-zero size");

    int rc = c3d_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "compress_auto decompress succeeded");

    double psnr = c3d_psnr(input, output, C3D_BLOCK_VOXELS);
    ASSERT(psnr > 20.0, "compress_auto roundtrip has reasonable PSNR");

    free(comp.data);
}

static void test_compress_roi(void) {
    uint8_t input[C3D_BLOCK_VOXELS];
    uint8_t output[C3D_BLOCK_VOXELS];
    fill_gradient(input);

    c3d_roi_t roi;
    roi.x0 = 8;  roi.y0 = 8;  roi.z0 = 8;
    roi.x1 = 24; roi.y1 = 24; roi.z1 = 24;
    roi.roi_quality = 95;

    /* Background at low quality, ROI at high quality */
    c3d_compressed_t comp = c3d_compress_roi(input, 20, &roi);
    ASSERT(comp.data != NULL, "compress_roi returned non-NULL");

    int rc = c3d_decompress(comp.data, comp.size, output);
    ASSERT(rc == 0, "compress_roi decompress succeeded");

    /* Compute error in ROI vs background */
    double roi_err = 0.0, bg_err = 0.0;
    int roi_count = 0, bg_count = 0;
    for (int z = 0; z < C3D_BLOCK_SIZE; z++) {
        for (int y = 0; y < C3D_BLOCK_SIZE; y++) {
            for (int x = 0; x < C3D_BLOCK_SIZE; x++) {
                int idx = z * C3D_BLOCK_SIZE * C3D_BLOCK_SIZE + y * C3D_BLOCK_SIZE + x;
                double diff = (double)input[idx] - (double)output[idx];
                double sq = diff * diff;
                if (x >= roi.x0 && x < roi.x1 &&
                    y >= roi.y0 && y < roi.y1 &&
                    z >= roi.z0 && z < roi.z1) {
                    roi_err += sq;
                    roi_count++;
                } else {
                    bg_err += sq;
                    bg_count++;
                }
            }
        }
    }
    roi_err /= roi_count;
    bg_err /= bg_count;
    ASSERT(roi_err < bg_err, "ROI region has lower MSE than background");

    free(comp.data);
}

static void test_deblock(void) {
    int vx = 64, vy = 64, vz = 64;
    size_t vol_size = (size_t)vx * vy * vz;
    uint8_t *volume = malloc(vol_size);

    /* Create smooth volume with artificial block boundary discontinuities.
     * Fill with a smooth gradient, then add jumps at chunk boundaries (every 32). */
    for (int z = 0; z < vz; z++) {
        for (int y = 0; y < vy; y++) {
            for (int x = 0; x < vx; x++) {
                int idx = z * vy * vx + y * vx + x;
                volume[idx] = (uint8_t)((x + y + z) % 200 + 28);
            }
        }
    }

    /* Add discontinuities at x=31/32 boundary */
    for (int z = 0; z < vz; z++) {
        for (int y = 0; y < vy; y++) {
            int idx31 = z * vy * vx + y * vx + 31;
            int idx32 = z * vy * vx + y * vx + 32;
            volume[idx31] = 200;
            volume[idx32] = 50;
        }
    }

    /* Measure boundary gradient before deblock */
    double grad_before = 0.0;
    int boundary_count = 0;
    for (int z = 0; z < vz; z++) {
        for (int y = 0; y < vy; y++) {
            int idx31 = z * vy * vx + y * vx + 31;
            int idx32 = z * vy * vx + y * vx + 32;
            grad_before += fabs((double)volume[idx31] - (double)volume[idx32]);
            boundary_count++;
        }
    }
    grad_before /= boundary_count;

    c3d_deblock(volume, vx, vy, vz, 1.0f);

    /* Measure boundary gradient after deblock */
    double grad_after = 0.0;
    for (int z = 0; z < vz; z++) {
        for (int y = 0; y < vy; y++) {
            int idx31 = z * vy * vx + y * vx + 31;
            int idx32 = z * vy * vx + y * vx + 32;
            grad_after += fabs((double)volume[idx31] - (double)volume[idx32]);
        }
    }
    grad_after /= boundary_count;

    ASSERT(grad_after < grad_before, "deblock reduced boundary gradients");
    ASSERT(grad_after < grad_before * 0.8, "deblock reduced gradients by at least 20%");

    free(volume);
}

/* Streaming API test state */
typedef struct {
    c3d_compressed_t chunks[16];
    int count;
} stream_state_t;

static void stream_write_cb(const uint8_t *data, size_t size, int chunk_index, void *userdata) {
    stream_state_t *state = (stream_state_t *)userdata;
    if (state->count < 16) {
        state->chunks[state->count].data = malloc(size);
        memcpy(state->chunks[state->count].data, data, size);
        state->chunks[state->count].size = size;
        state->count++;
    }
}

static void test_stream_compress(void) {
    stream_state_t state;
    memset(&state, 0, sizeof(state));

    c3d_stream_t *stream = c3d_stream_compress_create(80, 1, stream_write_cb, &state);
    ASSERT(stream != NULL, "stream_compress_create returned non-NULL");

    /* Push 4 blocks */
    uint8_t *blocks[4];
    for (int i = 0; i < 4; i++) {
        blocks[i] = malloc(C3D_BLOCK_VOXELS);
        for (int j = 0; j < C3D_BLOCK_VOXELS; j++)
            blocks[i][j] = (uint8_t)((j * (i + 3) + i * 17) % 256);
        int rc = c3d_stream_push(stream, blocks[i]);
        ASSERT(rc == 0, "stream_push succeeded");
    }

    int rc = c3d_stream_flush(stream);
    ASSERT(rc == 0, "stream_flush succeeded");

    ASSERT(state.count == 4, "stream produced 4 compressed outputs");

    /* Decompress each and verify */
    int all_ok = 1;
    for (int i = 0; i < state.count && i < 4; i++) {
        uint8_t output[C3D_BLOCK_VOXELS];
        rc = c3d_decompress(state.chunks[i].data, state.chunks[i].size, output);
        if (rc != 0) {
            all_ok = 0;
            break;
        }
        double psnr = c3d_psnr(blocks[i], output, C3D_BLOCK_VOXELS);
        if (psnr < 20.0) {
            all_ok = 0;
            break;
        }
    }
    ASSERT(all_ok, "stream compressed chunks decompress with reasonable quality");

    c3d_stream_free(stream);
    for (int i = 0; i < 4; i++)
        free(blocks[i]);
    for (int i = 0; i < state.count; i++)
        free(state.chunks[i].data);
}

static void test_null_safety(void) {
    uint8_t buf[C3D_BLOCK_VOXELS];

    /* Progressive */
    c3d_compressed_t comp = c3d_compress_progressive(NULL, 80);
    ASSERT(comp.data == NULL && comp.size == 0, "compress_progressive(NULL) returns {NULL,0}");
    ASSERT(c3d_decompress_progressive(NULL, 0, buf) != 0,
           "decompress_progressive(NULL) returns error");

    /* Meta */
    comp = c3d_compress_meta(NULL, 80, NULL);
    ASSERT(comp.data == NULL && comp.size == 0, "compress_meta(NULL) returns {NULL,0}");
    ASSERT(c3d_get_metadata(NULL, 0, NULL) != 0, "get_metadata(NULL) returns error");

    /* Target functions */
    comp = c3d_compress_target_psnr(NULL, 35.0);
    ASSERT(comp.data == NULL && comp.size == 0, "target_psnr(NULL) returns {NULL,0}");

    comp = c3d_compress_target_size(NULL, 8000);
    ASSERT(comp.data == NULL && comp.size == 0, "target_size(NULL) returns {NULL,0}");

    comp = c3d_compress_target_ssim(NULL, 0.9);
    ASSERT(comp.data == NULL && comp.size == 0, "target_ssim(NULL) returns {NULL,0}");

    /* Auto */
    comp = c3d_compress_auto(NULL, 80);
    ASSERT(comp.data == NULL && comp.size == 0, "compress_auto(NULL) returns {NULL,0}");

    /* ROI */
    comp = c3d_compress_roi(NULL, 80, NULL);
    ASSERT(comp.data == NULL && comp.size == 0, "compress_roi(NULL) returns {NULL,0}");

    /* Deblock: NULL should not crash */
    c3d_deblock(NULL, 64, 64, 64, 1.0f);
    passes++;

    /* Stream */
    ASSERT(c3d_stream_compress_create(80, 1, NULL, NULL) == NULL,
           "stream_compress_create(NULL cb) returns NULL");
    ASSERT(c3d_stream_push(NULL, buf) != 0, "stream_push(NULL) returns error");
    ASSERT(c3d_stream_flush(NULL) != 0, "stream_flush(NULL) returns error");
    c3d_stream_free(NULL); /* should not crash */
    passes++;
}

int main(void) {
    test_progressive_roundtrip();
    test_progressive_truncation();
    test_compress_meta();
    test_compress_target_psnr();
    test_compress_target_size();
    test_compress_target_ssim();
    test_compress_auto();
    test_compress_roi();
    test_deblock();
    test_stream_compress();
    test_null_safety();

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
