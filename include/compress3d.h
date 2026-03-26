#ifndef COMPRESS3D_H
#define COMPRESS3D_H

#include <stdint.h>
#include <stddef.h>

#define C3D_BLOCK_SIZE 32
#define C3D_BLOCK_VOXELS (C3D_BLOCK_SIZE * C3D_BLOCK_SIZE * C3D_BLOCK_SIZE)

/* Compressed data returned by c3d_compress. Caller must free `data`. */
typedef struct {
    uint8_t *data;
    size_t   size;
} c3d_compressed_t;

/*
 * Compress a 32^3 grayscale 8-bit volume.
 *   input:   32768 bytes, row-major (x fastest, then y, then z)
 *   quality: 1 (smallest/worst) to 100 (largest/best), 101 = lossless
 *            Clamped to [1, 101]. NULL input returns {NULL, 0}.
 * Returns compressed blob. Caller must free result.data.
 * On error, returns {NULL, 0}. Thread-safe (no shared state).
 */
c3d_compressed_t c3d_compress(const uint8_t *input, int quality);

/*
 * Decompress back to 32^3 grayscale 8-bit volume.
 *   compressed: blob from c3d_compress
 *   output:     must point to at least 32768 bytes
 * Returns 0 on success, -1 on error. NULL params return -1.
 * Thread-safe (no shared state). Equivalent to c3d_decompress_to.
 */
int c3d_decompress(const uint8_t *compressed, size_t compressed_size, uint8_t *output);

/*
 * Read the cube size from a compressed blob without decompressing.
 * Returns the size (8..256) on success, -1 on error. NULL-safe.
 */
int c3d_get_size(const uint8_t *compressed, size_t compressed_size);

/*
 * Returns the maximum possible compressed size for a 32^3 volume.
 * Caller can pre-allocate a buffer of this size.
 */
size_t c3d_compress_bound(void);

/*
 * Compress into a caller-provided buffer (always uses size=32).
 *   output must be at least c3d_compress_bound() bytes.
 *   quality: clamped to [1, 101]. NULL input/output returns 0.
 * Returns actual compressed size, or 0 on error. Thread-safe.
 */
size_t c3d_compress_to(const uint8_t *input, int quality, uint8_t *output, size_t output_cap);

/*
 * Decompress from compressed data into caller-provided output buffer.
 * Returns 0 on success, -1 on error. NULL params return -1.
 */
int c3d_decompress_to(const uint8_t *compressed, size_t compressed_size, uint8_t *output);

/* Compress a shard of multiple 32^3 chunks with inter-chunk DC delta coding.
 * chunks: array of pointers to 32768-byte volumes, in raster order (x fastest)
 * nx, ny, nz: number of chunks along each axis (e.g., 4,4,4 for 128^3)
 * quality: 1-101 (clamped). NULL chunks returns {NULL, 0}.
 * Returns compressed blob. Caller must free result.data.
 */
c3d_compressed_t c3d_compress_shard(const uint8_t **chunks, int nx, int ny, int nz, int quality, int num_threads);

/* Decompress a shard back to individual chunks.
 * chunks: array of pointers to pre-allocated 32768-byte buffers
 * Returns 0 on success, -1 on error. NULL params return -1.
 */
int c3d_decompress_shard(const uint8_t *compressed, size_t compressed_size,
                          uint8_t **chunks, int nx, int ny, int nz, int num_threads);

/* Compute 3D SSIM between two 32^3 volumes. Returns value in [0, 1].
 * NULL params return 0.0. Thread-safe. */
double c3d_ssim(const uint8_t *original, const uint8_t *reconstructed);

/* ══════════════════════════════════════════════════════════════════════════════
 * Quality metrics — all operate on arbitrary-length uint8 buffers.
 * Pass count = number of voxels. NULL params return 0.0 for doubles, 0 for ints.
 * c3d_psnr returns INFINITY for identical inputs. All are thread-safe.
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Mean Squared Error */
double c3d_mse(const uint8_t *a, const uint8_t *b, size_t count);

/* Peak Signal-to-Noise Ratio (dB). Returns INFINITY for identical inputs. */
double c3d_psnr(const uint8_t *a, const uint8_t *b, size_t count);

/* Mean Absolute Error */
double c3d_mae(const uint8_t *a, const uint8_t *b, size_t count);

/* Max Absolute Error (L-infinity norm) */
int c3d_max_error(const uint8_t *a, const uint8_t *b, size_t count);

/* Root Mean Squared Error */
double c3d_rmse(const uint8_t *a, const uint8_t *b, size_t count);

/* Normalized Root Mean Squared Error (NRMSE), normalized by value range. [0,1] */
double c3d_nrmse(const uint8_t *a, const uint8_t *b, size_t count);

/* Signal-to-Noise Ratio (dB). signal_var / noise_var. */
double c3d_snr(const uint8_t *original, const uint8_t *reconstructed, size_t count);

/* Pearson correlation coefficient. Returns value in [-1, 1]. */
double c3d_correlation(const uint8_t *a, const uint8_t *b, size_t count);

/* SSIM for arbitrary-length buffers (windowed, not just 32^3).
 * Uses 8^3 sliding window. Returns mean SSIM in [0, 1]. */
double c3d_ssim_volume(const uint8_t *a, const uint8_t *b,
                        int sx, int sy, int sz);

/* Multi-Scale SSIM (MS-SSIM). Downsamples 4 times, combines SSIM at each scale.
 * Returns value in [0, 1]. Requires dimensions >= 16. */
double c3d_ms_ssim(const uint8_t *a, const uint8_t *b,
                    int sx, int sy, int sz);

/* Histogram intersection: fraction of shared histogram area. [0, 1]. */
double c3d_histogram_intersection(const uint8_t *a, const uint8_t *b, size_t count);

/* Compression ratio: original_size / compressed_size. */
double c3d_compression_ratio(size_t original_size, size_t compressed_size);

/* Bits per voxel */
double c3d_bits_per_voxel(size_t compressed_size, size_t num_voxels);

/* Full quality report: computes all metrics at once. */
typedef struct {
    double mse;
    double psnr;
    double rmse;
    double nrmse;
    double mae;
    int    max_error;
    double snr;
    double ssim;
    double correlation;
    double histogram_intersection;
    double compression_ratio;
    double bits_per_voxel;
} c3d_quality_report_t;

/* Generate a full quality report comparing original vs reconstructed.
 * compressed_size: size of the compressed data (for ratio/bpv), 0 to skip.
 * Returns 0 on success, -1 on error. */
int c3d_quality_report(const uint8_t *original, const uint8_t *reconstructed,
                        size_t count, size_t compressed_size,
                        c3d_quality_report_t *report);

/* Print a quality report to stdout. */
void c3d_quality_report_print(const c3d_quality_report_t *report);

/* Opaque workspace (~264KB). Amortizes allocation across compress/decompress calls.
 * Thread safety: each thread needs its own workspace. Not safe to share. */
typedef struct c3d_workspace c3d_workspace_t;

/* Create a reusable workspace. Returns NULL on allocation failure. */
c3d_workspace_t *c3d_workspace_create(void);

/* Free a workspace. NULL-safe. */
void c3d_workspace_free(c3d_workspace_t *ws);

/* Compress using pre-allocated workspace (avoids internal malloc).
 * quality: clamped to [1, 101]. NULL params return 0. Not thread-safe for same ws. */
size_t c3d_compress_ws(const uint8_t *input, int quality, uint8_t *output, size_t output_cap, c3d_workspace_t *ws);

/* Decompress using pre-allocated workspace.
 * NULL params return -1. Not thread-safe for same ws. */
int c3d_decompress_ws(const uint8_t *compressed, size_t compressed_size, uint8_t *output, c3d_workspace_t *ws);

/* Metadata for medical/scientific volumes */
typedef struct {
    float voxel_size[3];     /* Voxel dimensions in mm (0 = unspecified) */
    float origin[3];         /* Volume origin in world coordinates */
    uint16_t modality;       /* 0=unknown, 1=CT, 2=MRI, 3=XRay */
    uint16_t bits_per_voxel; /* Original bit depth (8 for our case) */
} c3d_metadata_t;

/*
 * Compress with metadata and CRC32 integrity check.
 * meta may be NULL for no metadata (CRC is still added).
 * quality: clamped to [1, 101]. NULL input returns {NULL, 0}.
 */
c3d_compressed_t c3d_compress_meta(const uint8_t *input, int quality, const c3d_metadata_t *meta);

/*
 * Read metadata from a compressed blob without decompressing.
 * Returns 0 on success, -1 if no metadata present or format error.
 */
int c3d_get_metadata(const uint8_t *compressed, size_t size, c3d_metadata_t *meta);

/*
 * Compress using CDF 5/3 (Le Gall) wavelet transform instead of DCT.
 * Better for sharp edges (less ringing). Integer-based, faster than DCT.
 * Same interface as c3d_compress. Decompress auto-detects transform type.
 * quality: clamped to [1, 101]. NULL input returns {NULL, 0}. Thread-safe.
 */
c3d_compressed_t c3d_compress_wavelet(const uint8_t *input, int quality);

/*
 * Compress with progressive bitstream. Can be truncated at any byte
 * boundary for progressive decode (lower quality but valid output).
 * Coefficients are encoded in zigzag order using raw VLQ+RLE (no rANS).
 * quality: clamped to [1, 101]. NULL input returns {NULL, 0}. Thread-safe.
 */
c3d_compressed_t c3d_compress_progressive(const uint8_t *input, int quality);

/*
 * Decompress progressive bitstream. Handles truncated data gracefully —
 * missing coefficients are treated as zero. Returns 0 on success, -1 on error.
 */
int c3d_decompress_progressive(const uint8_t *compressed, size_t compressed_size, uint8_t *output);

/* Compress with automatic transform selection (DCT or wavelet).
 * Analyzes edge strength to pick the best transform per chunk.
 * High edge strength -> wavelet (less ringing), low -> DCT (better compaction).
 * quality: clamped to [1, 101]. NULL input returns {NULL, 0}. Thread-safe. */
c3d_compressed_t c3d_compress_auto(const uint8_t *input, int quality);

/* Compress to a target PSNR (dB). Binary searches quality 1-100.
 * NULL input returns {NULL, 0}. Thread-safe. */
c3d_compressed_t c3d_compress_target_psnr(const uint8_t *input, double target_psnr);

/* Compress to a target file size (bytes). Binary searches quality 1-100.
 * NULL input returns {NULL, 0}. Thread-safe. */
c3d_compressed_t c3d_compress_target_size(const uint8_t *input, size_t target_size);

/* Compress to a target SSIM (0.0-1.0). Binary searches quality 1-100.
 * NULL input returns {NULL, 0}. Thread-safe. */
c3d_compressed_t c3d_compress_target_ssim(const uint8_t *input, double target_ssim);

/* Decompress a single chunk from a shard by chunk index (0-based, raster order).
 * output: must point to at least 32768 bytes.
 * Returns 0 on success, -1 on error.
 */
int c3d_decompress_shard_chunk(const uint8_t *compressed, size_t compressed_size,
                                int chunk_index, uint8_t *output);

/* Get the number of chunks in a shard. Returns count or -1 on error. */
int c3d_shard_chunk_count(const uint8_t *compressed, size_t compressed_size);

/* ── Memory-mapped shard access (POSIX only) ── */
#ifndef _WIN32

/* Opaque handle for a memory-mapped shard file. */
typedef struct c3d_shard_map c3d_shard_map_t;

/* Open a shard file with mmap for random-access chunk reading.
 * Returns handle on success, NULL on error. */
c3d_shard_map_t *c3d_shard_mmap_open(const char *path);

/* Get chunk count from mapped shard. */
int c3d_shard_mmap_chunk_count(const c3d_shard_map_t *map);

/* Decompress a single chunk from the mapped shard.
 * output must point to at least 32768 bytes. Returns 0 on success. */
int c3d_shard_mmap_read_chunk(const c3d_shard_map_t *map, int chunk_index, uint8_t *output);

/* Close and unmap. */
void c3d_shard_mmap_close(c3d_shard_map_t *map);

#endif /* !_WIN32 */

/* ── Streaming shard writer ── */

typedef struct c3d_shard_writer c3d_shard_writer_t;

/* Begin writing a shard. Writes header placeholder. */
c3d_shard_writer_t *c3d_shard_writer_open(const char *path, int nx, int ny, int nz, int quality);

/* Add the next chunk (must be added in raster order). Returns 0 on success. */
int c3d_shard_writer_add_chunk(c3d_shard_writer_t *w, const uint8_t *chunk);

/* Finalize: write the offset table and close. Returns 0 on success. */
int c3d_shard_writer_finish(c3d_shard_writer_t *w);

/* ── Streaming API ── */

/* Callback for receiving compressed chunk data. */
typedef void (*c3d_write_cb)(const uint8_t *data, size_t size, int chunk_index, void *userdata);

/* Callback for receiving decompressed chunk data. */
typedef void (*c3d_read_cb)(uint8_t *output, int chunk_index, void *userdata);

typedef struct c3d_stream c3d_stream_t;

/* Create a streaming compressor.
 * Compresses chunks as they arrive via c3d_stream_push.
 * Calls write_cb for each compressed chunk.
 * quality: clamped to [1, 101]. NULL write_cb returns NULL.
 * num_threads: worker thread count (0 = auto)
 */
c3d_stream_t *c3d_stream_compress_create(int quality, int num_threads,
                                          c3d_write_cb write_cb, void *userdata);

/* Push a 32^3 chunk into the stream. Non-blocking if worker threads available.
 * Returns 0 on success, -1 on error. */
int c3d_stream_push(c3d_stream_t *stream, const uint8_t *chunk);

/* Flush remaining chunks and wait for completion. */
int c3d_stream_flush(c3d_stream_t *stream);

/* Destroy stream. */
void c3d_stream_free(c3d_stream_t *stream);

/* ── ROI (Region of Interest) coding ── */

typedef struct {
    int x0, y0, z0;  /* ROI start (inclusive) */
    int x1, y1, z1;  /* ROI end (exclusive) */
    int roi_quality;  /* Quality for the ROI region (1-100) */
} c3d_roi_t;

/* Compress with ROI: background at `quality`, ROI region at `roi.roi_quality`.
 * quality: clamped to [1, 101]. NULL input/roi returns {NULL, 0}.
 * Returns compressed blob. Caller must free result.data.
 */
c3d_compressed_t c3d_compress_roi(const uint8_t *input, int quality, const c3d_roi_t *roi);

/* Apply 3D deblocking filter along chunk boundaries.
 * volume: pointer to the full assembled volume (all chunks combined)
 * vx, vy, vz: full volume dimensions in voxels (must be multiples of 32)
 * strength: 0.0 (no filtering) to 1.0 (maximum smoothing)
 * Only smooths across 32-voxel chunk boundaries, not within chunks.
 */
void c3d_deblock(uint8_t *volume, int vx, int vy, int vz, float strength);

/* ══════════════════════════════════════════════════════════════════════════════
 * Multiscale Pyramid — "C3M\x01" container
 *
 * Stores a full power-of-2 resolution pyramid where each level encodes only
 * the residual detail relative to the upsampled coarser level. Level 0 is the
 * coarsest (1x1x1 = single voxel), level N is native resolution.
 *
 * Decoding is progressive: to reconstruct level K you need levels 0..K-1
 * (cached), upsample K-1 to K dimensions, and add the level K residuals.
 * ══════════════════════════════════════════════════════════════════════════════ */

#define C3D_MAX_LEVELS 20
#define C3D_UPSAMPLE_NEAREST   0
#define C3D_UPSAMPLE_TRILINEAR 1
#define C3D_UPSAMPLE_CUBIC     2

/* Multiscale compression parameters. */
typedef struct {
    int quality;          /* 1-101 (same as c3d_compress) */
    int upsample_method;  /* C3D_UPSAMPLE_TRILINEAR recommended */
    int num_threads;      /* 0 = auto */
} c3d_multiscale_params_t;

/* Per-level metadata in a C3M container. */
typedef struct {
    uint32_t offset;      /* byte offset from container start */
    uint32_t size;        /* compressed size */
    uint16_t dim_x, dim_y, dim_z;
} c3d_level_info_t;

/* Header for a C3M container. */
typedef struct {
    int num_levels;
    int quality;
    int upsample_method;
    uint32_t native_x, native_y, native_z;
    uint32_t total_size;
    c3d_level_info_t levels[C3D_MAX_LEVELS];
} c3d_multiscale_header_t;

/* Downsample a volume by 2x in each dimension (2x2x2 average pooling).
 * input: sx*sy*sz bytes. output: (sx/2)*(sy/2)*(sz/2) bytes.
 * sx, sy, sz must be even. Returns 0 on success, -1 on error. */
int c3d_downsample_2x(const uint8_t *input, int sx, int sy, int sz, uint8_t *output);

/* Upsample a volume by 2x in each dimension.
 * input: sx*sy*sz bytes. output: (2*sx)*(2*sy)*(2*sz) bytes.
 * method: C3D_UPSAMPLE_NEAREST, C3D_UPSAMPLE_TRILINEAR, or C3D_UPSAMPLE_CUBIC.
 * Returns 0 on success, -1 on error. */
int c3d_upsample_2x(const uint8_t *input, int sx, int sy, int sz,
                     int method, uint8_t *output);

/* Compress a volume into a multiscale C3M container.
 * input: native_x * native_y * native_z bytes (dimensions must be powers of 2).
 * params: compression parameters.
 * Returns compressed blob. Caller must free result.data. */
c3d_compressed_t c3d_multiscale_compress(const uint8_t *input,
    int native_x, int native_y, int native_z,
    const c3d_multiscale_params_t *params);

/* Read the header from a C3M container without decompressing.
 * Returns 0 on success, -1 on error. */
int c3d_multiscale_header(const uint8_t *data, size_t size,
                           c3d_multiscale_header_t *header);

/* Opaque cache for decoded pyramid levels. */
typedef struct c3d_level_cache c3d_level_cache_t;

/* Create a level cache. Returns NULL on failure. */
c3d_level_cache_t *c3d_level_cache_create(void);

/* Free a level cache and all cached levels. */
void c3d_level_cache_free(c3d_level_cache_t *cache);

/* Decompress a single level from a C3M container.
 * Recursively decodes coarser levels as needed, using the cache.
 * output: must point to level_dimx * level_dimy * level_dimz bytes.
 * Returns 0 on success, -1 on error. */
int c3d_multiscale_decompress_level(const uint8_t *data, size_t size,
                                     int target_level,
                                     uint8_t *output,
                                     c3d_level_cache_t *cache);

/* Convenience: decompress at native resolution (highest level).
 * output must be native_x * native_y * native_z bytes. */
int c3d_multiscale_decompress(const uint8_t *data, size_t size,
                               uint8_t *output);

/* ══════════════════════════════════════════════════════════════════════════════
 * UDP Serving Protocol
 *
 * Streams multiscale C3D data over UDP with coarse-to-fine progressive
 * delivery. Designed for low-latency volumetric data viewing.
 * ══════════════════════════════════════════════════════════════════════════════ */

#ifndef _WIN32

#define C3D_UDP_PORT_DEFAULT 7333

/* Server */
typedef struct c3d_server c3d_server_t;

/* Create a UDP server for a C3M volume (or directory of shards).
 * volume_path: path to C3M file or shard directory.
 * port: UDP port to bind (0 = C3D_UDP_PORT_DEFAULT). */
c3d_server_t *c3d_server_create(const char *volume_path, int port);

/* Run the server event loop (blocks). Returns 0 on clean shutdown. */
int c3d_server_run(c3d_server_t *server);

/* Signal the server to stop. Thread-safe. */
void c3d_server_stop(c3d_server_t *server);

/* Free server resources. */
void c3d_server_free(c3d_server_t *server);

/* Client */
typedef struct c3d_client c3d_client_t;

/* Callback when a decompressed chunk arrives. */
typedef void (*c3d_chunk_cb)(int level, int cx, int cy, int cz,
                              const uint8_t *data, size_t data_size,
                              void *userdata);

/* Create a UDP client connecting to a C3D server. */
c3d_client_t *c3d_client_create(const char *host, int port,
                                 c3d_chunk_cb cb, void *userdata);

/* Request a box region at up to max_level detail.
 * min_level: skip coarser levels if already cached.
 * Chunks are delivered via the callback, coarse-to-fine. */
int c3d_client_request_region(c3d_client_t *c,
    int x0, int y0, int z0, int x1, int y1, int z1,
    int max_level, int min_level, int priority);

/* Cancel pending requests above a level. */
int c3d_client_cancel_above(c3d_client_t *c, int level);

/* Poll for incoming data. timeout_ms: -1=block, 0=non-blocking.
 * Returns number of chunks received, or -1 on error. */
int c3d_client_poll(c3d_client_t *c, int timeout_ms);

/* Destroy client. */
void c3d_client_free(c3d_client_t *c);

#endif /* !_WIN32 */

#endif /* COMPRESS3D_H */
