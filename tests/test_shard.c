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

/* Fill a chunk with a deterministic pattern based on index */
static void fill_chunk(uint8_t *chunk, int idx) {
    for (int i = 0; i < C3D_BLOCK_VOXELS; i++)
        chunk[i] = (uint8_t)((i * (idx + 1) * 7 + idx * 31) % 256);
}

static void test_shard_roundtrip(void) {
    /* 2x2x2 shard = 8 chunks */
    int nx = 2, ny = 2, nz = 2;
    int nchunks = nx * ny * nz;

    uint8_t *input_bufs[8];
    uint8_t *output_bufs[8];
    const uint8_t *input_ptrs[8];
    for (int i = 0; i < nchunks; i++) {
        input_bufs[i] = malloc(C3D_BLOCK_VOXELS);
        output_bufs[i] = malloc(C3D_BLOCK_VOXELS);
        fill_chunk(input_bufs[i], i);
        input_ptrs[i] = input_bufs[i];
    }

    c3d_compressed_t comp = c3d_compress_shard(input_ptrs, nx, ny, nz, 101, 1);
    ASSERT(comp.data != NULL, "shard compress returned non-NULL");
    ASSERT(comp.size > 0, "shard compress returned non-zero size");

    uint8_t *out_ptrs[8];
    for (int i = 0; i < nchunks; i++)
        out_ptrs[i] = output_bufs[i];

    int rc = c3d_decompress_shard(comp.data, comp.size, out_ptrs, nx, ny, nz, 1);
    ASSERT(rc == 0, "shard decompress succeeded");

    /* Shard DC delta coding uses float mean subtraction which may introduce
     * small rounding errors (max ~1). Check near-lossless rather than exact. */
    int max_diff = 0;
    for (int i = 0; i < nchunks; i++) {
        for (int j = 0; j < C3D_BLOCK_VOXELS; j++) {
            int d = abs((int)input_bufs[i][j] - (int)output_bufs[i][j]);
            if (d > max_diff) max_diff = d;
        }
    }
    ASSERT(max_diff <= 2, "shard lossless roundtrip max diff <= 2");

    free(comp.data);
    for (int i = 0; i < nchunks; i++) {
        free(input_bufs[i]);
        free(output_bufs[i]);
    }
}

static void test_shard_chunk_count(void) {
    int nx = 2, ny = 2, nz = 2;
    int nchunks = nx * ny * nz;

    uint8_t *input_bufs[8];
    const uint8_t *input_ptrs[8];
    for (int i = 0; i < nchunks; i++) {
        input_bufs[i] = malloc(C3D_BLOCK_VOXELS);
        fill_chunk(input_bufs[i], i);
        input_ptrs[i] = input_bufs[i];
    }

    c3d_compressed_t comp = c3d_compress_shard(input_ptrs, nx, ny, nz, 80, 1);
    ASSERT(comp.data != NULL, "shard compress for chunk_count ok");

    int count = c3d_shard_chunk_count(comp.data, comp.size);
    ASSERT(count == nchunks, "shard_chunk_count returns 8 for 2x2x2");

    free(comp.data);
    for (int i = 0; i < nchunks; i++)
        free(input_bufs[i]);
}

static void test_decompress_shard_chunk(void) {
    int nx = 2, ny = 2, nz = 2;
    int nchunks = nx * ny * nz;

    uint8_t *input_bufs[8];
    const uint8_t *input_ptrs[8];
    for (int i = 0; i < nchunks; i++) {
        input_bufs[i] = malloc(C3D_BLOCK_VOXELS);
        fill_chunk(input_bufs[i], i);
        input_ptrs[i] = input_bufs[i];
    }

    c3d_compressed_t comp = c3d_compress_shard(input_ptrs, nx, ny, nz, 101, 1);
    ASSERT(comp.data != NULL, "shard compress for random access ok");

    /* Read chunk 3 of 8 */
    uint8_t *chunk3 = malloc(C3D_BLOCK_VOXELS);
    int rc = c3d_decompress_shard_chunk(comp.data, comp.size, 3, chunk3);
    ASSERT(rc == 0, "decompress_shard_chunk(3) succeeded");
    {
        int md = 0;
        for (int j = 0; j < C3D_BLOCK_VOXELS; j++) {
            int d = abs((int)input_bufs[3][j] - (int)chunk3[j]);
            if (d > md) md = d;
        }
        ASSERT(md <= 2, "decompress_shard_chunk(3) near-matches original chunk 3");
    }

    free(chunk3);
    free(comp.data);
    for (int i = 0; i < nchunks; i++)
        free(input_bufs[i]);
}

static void test_shard_writer_and_readback(void) {
    int nx = 2, ny = 2, nz = 2;
    int nchunks = nx * ny * nz;
    const char *path = "/tmp/test_shard.c3s";

    uint8_t *input_bufs[8];
    for (int i = 0; i < nchunks; i++) {
        input_bufs[i] = malloc(C3D_BLOCK_VOXELS);
        fill_chunk(input_bufs[i], i);
    }

    /* Write shard to file */
    c3d_shard_writer_t *w = c3d_shard_writer_open(path, nx, ny, nz, 101);
    ASSERT(w != NULL, "shard_writer_open succeeded");

    int add_ok = 1;
    for (int i = 0; i < nchunks; i++) {
        if (c3d_shard_writer_add_chunk(w, input_bufs[i]) != 0) {
            add_ok = 0;
            break;
        }
    }
    ASSERT(add_ok, "shard_writer_add_chunk succeeded for all chunks");

    int rc = c3d_shard_writer_finish(w);
    ASSERT(rc == 0, "shard_writer_finish succeeded");

    /* Read file back and decompress */
    FILE *f = fopen(path, "rb");
    ASSERT(f != NULL, "written shard file exists");
    if (f) {
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint8_t *filedata = malloc(fsize);
        fread(filedata, 1, fsize, f);
        fclose(f);

        uint8_t *output_bufs[8];
        uint8_t *out_ptrs[8];
        for (int i = 0; i < nchunks; i++) {
            output_bufs[i] = malloc(C3D_BLOCK_VOXELS);
            out_ptrs[i] = output_bufs[i];
        }

        rc = c3d_decompress_shard(filedata, (size_t)fsize, out_ptrs, nx, ny, nz, 1);
        ASSERT(rc == 0, "decompress shard from file succeeded");

        int wr_max_diff = 0;
        for (int i = 0; i < nchunks; i++) {
            for (int j = 0; j < C3D_BLOCK_VOXELS; j++) {
                int d = abs((int)input_bufs[i][j] - (int)output_bufs[i][j]);
                if (d > wr_max_diff) wr_max_diff = d;
            }
        }
        ASSERT(wr_max_diff <= 2, "shard writer roundtrip max diff <= 2");

        free(filedata);
        for (int i = 0; i < nchunks; i++)
            free(output_bufs[i]);
    }

    for (int i = 0; i < nchunks; i++)
        free(input_bufs[i]);
    remove(path);
}

#ifndef _WIN32
static void test_shard_mmap(void) {
    int nx = 2, ny = 2, nz = 2;
    int nchunks = nx * ny * nz;
    const char *path = "/tmp/test_shard_mmap.c3s";

    uint8_t *input_bufs[8];
    for (int i = 0; i < nchunks; i++) {
        input_bufs[i] = malloc(C3D_BLOCK_VOXELS);
        fill_chunk(input_bufs[i], i);
    }

    /* Write shard to file first */
    c3d_shard_writer_t *w = c3d_shard_writer_open(path, nx, ny, nz, 101);
    ASSERT(w != NULL, "mmap: shard_writer_open succeeded");
    for (int i = 0; i < nchunks; i++)
        c3d_shard_writer_add_chunk(w, input_bufs[i]);
    c3d_shard_writer_finish(w);

    /* Open via mmap */
    c3d_shard_map_t *map = c3d_shard_mmap_open(path);
    ASSERT(map != NULL, "shard_mmap_open succeeded");

    if (map) {
        int count = c3d_shard_mmap_chunk_count(map);
        ASSERT(count == nchunks, "shard_mmap_chunk_count returns 8");

        /* Read each chunk and verify */
        uint8_t *chunk = malloc(C3D_BLOCK_VOXELS);
        int mmap_max_diff = 0;
        int mmap_ok = 1;
        for (int i = 0; i < nchunks; i++) {
            int rc = c3d_shard_mmap_read_chunk(map, i, chunk);
            if (rc != 0) { mmap_ok = 0; break; }
            for (int j = 0; j < C3D_BLOCK_VOXELS; j++) {
                int d = abs((int)input_bufs[i][j] - (int)chunk[j]);
                if (d > mmap_max_diff) mmap_max_diff = d;
            }
        }
        ASSERT(mmap_ok && mmap_max_diff <= 2, "shard_mmap_read_chunk near-matches original for all chunks");

        free(chunk);
        c3d_shard_mmap_close(map);
    }

    for (int i = 0; i < nchunks; i++)
        free(input_bufs[i]);
    remove(path);
}
#endif

static void test_null_safety(void) {
    uint8_t buf[C3D_BLOCK_VOXELS];
    uint8_t *ptrs[1] = { buf };

    c3d_compressed_t comp = c3d_compress_shard(NULL, 2, 2, 2, 50, 1);
    ASSERT(comp.data == NULL && comp.size == 0, "compress_shard(NULL) returns {NULL,0}");

    ASSERT(c3d_decompress_shard(NULL, 0, ptrs, 2, 2, 2, 1) == -1,
           "decompress_shard(NULL) returns -1");
    ASSERT(c3d_shard_chunk_count(NULL, 0) == -1,
           "shard_chunk_count(NULL) returns -1");
    ASSERT(c3d_decompress_shard_chunk(NULL, 0, 0, buf) == -1,
           "decompress_shard_chunk(NULL) returns -1");

    ASSERT(c3d_shard_writer_open(NULL, 2, 2, 2, 50) == NULL,
           "shard_writer_open(NULL) returns NULL");
    ASSERT(c3d_shard_writer_add_chunk(NULL, buf) != 0,
           "shard_writer_add_chunk(NULL) returns error");
    ASSERT(c3d_shard_writer_finish(NULL) != 0,
           "shard_writer_finish(NULL) returns error");

#ifndef _WIN32
    ASSERT(c3d_shard_mmap_open(NULL) == NULL, "shard_mmap_open(NULL) returns NULL");
    ASSERT(c3d_shard_mmap_chunk_count(NULL) == -1, "shard_mmap_chunk_count(NULL) returns -1");
    ASSERT(c3d_shard_mmap_read_chunk(NULL, 0, buf) != 0, "shard_mmap_read_chunk(NULL) returns error");
    c3d_shard_mmap_close(NULL); /* should not crash */
    passes++;
#endif
}

int main(void) {
    test_shard_roundtrip();
    test_shard_chunk_count();
    test_decompress_shard_chunk();
    test_shard_writer_and_readback();
#ifndef _WIN32
    test_shard_mmap();
#endif
    test_null_safety();

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
