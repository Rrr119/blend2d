// This file is part of Blend2D project <https://blend2d.com>
//
// See blend2d.h or LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

#include <blend2d/core/api-build_p.h>
#include <blend2d/core/imagefilter_p.h>
#include <blend2d/core/runtime_p.h>

namespace bl {
namespace ImageFilterInternal {

// bl::ImageFilterInternal - Constants
// ===================================

static constexpr int kMaxBlurRadius = 8192;
static constexpr uint32_t kMaxBoxPasses = 8;

static constexpr uint32_t kChannelShift[4] = { 16u, 8u, 0u, 24u };

struct ResolvedBlur {
  double radius;
  double sigma;
  int kernel_radius;
};

// bl::ImageFilterInternal - Common Helpers
// ========================================

static BL_INLINE bool is_filter_format(BLFormat format) noexcept {
  return format == BL_FORMAT_PRGB32 || format == BL_FORMAT_XRGB32 || format == BL_FORMAT_A8;
}

static BL_INLINE bool is_valid_blur_flags(uint32_t flags) noexcept {
  return (flags & ~uint32_t(BL_BLUR_FLAG_TRANSPARENT_EDGE | BL_BLUR_FLAG_APPROXIMATE)) == 0;
}

static BL_INLINE bool is_valid_shadow_flags(uint32_t flags) noexcept {
  return (flags & ~uint32_t(BL_SHADOW_FLAG_SHADOW_ONLY | BL_SHADOW_FLAG_APPROXIMATE)) == 0;
}

static BLResult resolve_blur(const BLBlurOptions* options, ResolvedBlur* resolved) noexcept {
  if (!std::isfinite(options->radius) || !std::isfinite(options->sigma))
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  if (options->radius < 0.0 || options->sigma < 0.0)
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  double radius = options->radius;
  double sigma = options->sigma;

  if (radius == 0.0 && sigma == 0.0) {
    resolved->radius = 0.0;
    resolved->sigma = 0.0;
    resolved->kernel_radius = 0;
    return BL_SUCCESS;
  }

  if (sigma == 0.0)
    sigma = radius / 3.0;

  if (radius == 0.0)
    radius = std::ceil(3.0 * sigma);
  else
    radius = std::ceil(radius);

  if (!(sigma > 0.0) || !std::isfinite(radius) || radius > double(kMaxBlurRadius))
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  resolved->radius = radius;
  resolved->sigma = sigma;
  resolved->kernel_radius = int(radius);
  return BL_SUCCESS;
}

static BLResult resolve_box_blur(const BLBlurOptions* options, ResolvedBlur* resolved, uint32_t* passes_out) noexcept {
  BL_PROPAGATE(resolve_blur(options, resolved));

  uint32_t passes = options->passes ? options->passes : 1u;
  if (passes > kMaxBoxPasses)
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  *passes_out = passes;
  return BL_SUCCESS;
}

static BL_INLINE EdgeMode edge_mode_from_flags(uint32_t flags) noexcept {
  return (flags & BL_BLUR_FLAG_TRANSPARENT_EDGE) ? EdgeMode::kZero : EdgeMode::kClamp;
}

static BLResult check_plane_size(int w, int h, size_t* size_out) noexcept {
  uint64_t size = uint64_t(uint32_t(w)) * uint64_t(uint32_t(h));
  if (size > SIZE_MAX)
    return bl_make_error(BL_ERROR_IMAGE_TOO_LARGE);
  *size_out = size_t(size);
  return BL_SUCCESS;
}

template<typename T>
static BL_INLINE BLResult alloc_array(BLArray<T>& array, size_t size, T** data_out) noexcept {
  BL_PROPAGATE(array.resize(size, T(0)));
  return array.make_mutable(data_out);
}

static BLResult copy_plane(uint8_t* dst_data, intptr_t dst_stride, const uint8_t* src_data, intptr_t src_stride, int w, int h) noexcept {
  for (int y = 0; y < h; y++)
    memcpy(dst_data + intptr_t(y) * dst_stride, src_data + intptr_t(y) * src_stride, size_t(w));
  return BL_SUCCESS;
}

// bl::ImageFilterInternal - Blur Dispatch
// =======================================

static BLResult box_blur_plane_pass(
  uint8_t* dst_data, intptr_t dst_stride,
  const uint8_t* src_data, intptr_t src_stride,
  int w, int h, int radius, EdgeMode edge_mode) noexcept {

  if (radius <= 0)
    return copy_plane(dst_data, dst_stride, src_data, src_stride, w, h);

#if defined(BL_BUILD_OPT_AVX2)
  if (radius <= 128 && bl_runtime_has_avx2(&bl_runtime_context))
    return box_blur_plane_avx2(dst_data, dst_stride, src_data, src_stride, w, h, radius, edge_mode);
#endif

#if defined(BL_BUILD_OPT_SSE2)
  if (radius <= 128 && bl_runtime_has_sse2(&bl_runtime_context))
    return box_blur_plane_sse2(dst_data, dst_stride, src_data, src_stride, w, h, radius, edge_mode);
#endif

  return box_blur_plane_scalar(dst_data, dst_stride, src_data, src_stride, w, h, radius, edge_mode);
}

static BLResult box_blur_plane_multi(
  uint8_t* dst_data, intptr_t dst_stride,
  const uint8_t* src_data, intptr_t src_stride,
  int w, int h, int radius, uint32_t passes, EdgeMode edge_mode) noexcept {

  if (passes == 1)
    return box_blur_plane_pass(dst_data, dst_stride, src_data, src_stride, w, h, radius, edge_mode);

  size_t plane_size;
  BL_PROPAGATE(check_plane_size(w, h, &plane_size));

  BLArray<uint8_t> scratch_array;
  uint8_t* scratch_data;
  BL_PROPAGATE(alloc_array(scratch_array, plane_size, &scratch_data));

  const uint8_t* current_data = src_data;
  intptr_t current_stride = src_stride;

  for (uint32_t pass = 0; pass < passes; pass++) {
    bool last = pass + 1 == passes;
    uint8_t* target_data = last ? dst_data : scratch_data;
    intptr_t target_stride = last ? dst_stride : intptr_t(w);

    BL_PROPAGATE(box_blur_plane_pass(target_data, target_stride, current_data, current_stride, w, h, radius, edge_mode));

    current_data = target_data;
    current_stride = target_stride;
  }

  return BL_SUCCESS;
}

static void compute_box_radii(double sigma, int radius_cap, int radii[3]) noexcept {
  constexpr double n = 3.0;
  sigma = bl_min(sigma, double(bl_max(radius_cap, 1)));
  double sigma2 = sigma * sigma;

  double ideal = std::sqrt(12.0 * sigma2 / n + 1.0);
  int wl = int(std::floor(ideal));
  if ((wl & 1) == 0)
    wl--;
  if (wl < 1)
    wl = 1;

  int wu = wl + 2;
  double m_ideal = (12.0 * sigma2 - n * double(wl) * double(wl) - 4.0 * n * double(wl) - 3.0 * n) / (-4.0 * double(wl) - 4.0);
  int m = int(m_ideal + 0.5);

  if (m < 0)
    m = 0;
  if (m > 3)
    m = 3;

  for (int i = 0; i < 3; i++) {
    int box_width = i < m ? wl : wu;
    int radius = (box_width - 1) / 2;

    if (radius > radius_cap)
      radius = radius_cap;
    if (radius < 0)
      radius = 0;

    radii[i] = radius;
  }
}

static BLResult gaussian_blur_plane(
  uint8_t* dst_data, intptr_t dst_stride,
  const uint8_t* src_data, intptr_t src_stride,
  int w, int h, double radius, double sigma, EdgeMode edge_mode, bool approximate) noexcept {

  if (approximate) {
    int radii[3];
    compute_box_radii(sigma, int(radius), radii);

    size_t plane_size;
    BL_PROPAGATE(check_plane_size(w, h, &plane_size));

    BLArray<uint8_t> scratch_array;
    uint8_t* scratch_data;
    BL_PROPAGATE(alloc_array(scratch_array, plane_size, &scratch_data));

    const uint8_t* current_data = src_data;
    intptr_t current_stride = src_stride;

    for (int i = 0; i < 3; i++) {
      bool last = i == 2;
      uint8_t* target_data = last ? dst_data : scratch_data;
      intptr_t target_stride = last ? dst_stride : intptr_t(w);

      BL_PROPAGATE(box_blur_plane_pass(target_data, target_stride, current_data, current_stride, w, h, radii[i], edge_mode));

      current_data = target_data;
      current_stride = target_stride;
    }

    return BL_SUCCESS;
  }

  int kernel_radius = int(radius);
  size_t kernel_size = size_t(kernel_radius) * 2u + 1u;

  BLArray<float> kernel_array;
  float* kernel;
  BL_PROPAGATE(alloc_array(kernel_array, kernel_size, &kernel));

  double sigma2 = sigma * sigma;
  double sum = 0.0;

  for (int i = -kernel_radius; i <= kernel_radius; i++) {
    double x = double(i);
    double weight = std::exp(-0.5 * x * x / sigma2);
    kernel[i + kernel_radius] = float(weight);
    sum += weight;
  }

  for (size_t i = 0; i < kernel_size; i++)
    kernel[i] = float(double(kernel[i]) / sum);

  return gaussian_blur_plane_scalar(dst_data, dst_stride, src_data, src_stride, w, h, kernel, kernel_radius, edge_mode);
}

// bl::ImageFilterInternal - Plane Access
// ======================================

static void extract_plane(
  uint8_t* plane_data,
  const uint8_t* image_data, intptr_t image_stride,
  int w, int h, uint32_t format, uint32_t channel) noexcept {

  if (format == BL_FORMAT_A8) {
    for (int y = 0; y < h; y++)
      memcpy(plane_data + size_t(y) * size_t(w), image_data + intptr_t(y) * image_stride, size_t(w));
    return;
  }

  uint32_t shift = kChannelShift[channel];
  bool is_opaque_channel = format == BL_FORMAT_XRGB32 && channel == 3;

  for (int y = 0; y < h; y++) {
    const uint32_t* src_row = reinterpret_cast<const uint32_t*>(image_data + intptr_t(y) * image_stride);
    uint8_t* dst_row = plane_data + size_t(y) * size_t(w);

    if (is_opaque_channel) {
      memset(dst_row, 0xFF, size_t(w));
    }
    else {
      for (int x = 0; x < w; x++)
        dst_row[x] = uint8_t((src_row[x] >> shift) & 0xFFu);
    }
  }
}

static void store_plane(
  uint8_t* image_data, intptr_t image_stride,
  const uint8_t* plane_data,
  int w, int h, uint32_t format, uint32_t channel) noexcept {

  if (format == BL_FORMAT_A8) {
    for (int y = 0; y < h; y++)
      memcpy(image_data + intptr_t(y) * image_stride, plane_data + size_t(y) * size_t(w), size_t(w));
    return;
  }

  uint32_t shift = kChannelShift[channel];
  uint32_t mask = uint32_t(0xFF) << shift;
  bool is_opaque_channel = format == BL_FORMAT_XRGB32 && channel == 3;

  for (int y = 0; y < h; y++) {
    uint32_t* dst_row = reinterpret_cast<uint32_t*>(image_data + intptr_t(y) * image_stride);
    const uint8_t* src_row = plane_data + size_t(y) * size_t(w);

    if (is_opaque_channel) {
      for (int x = 0; x < w; x++)
        dst_row[x] |= 0xFF000000u;
    }
    else {
      for (int x = 0; x < w; x++)
        dst_row[x] = (dst_row[x] & ~mask) | (uint32_t(src_row[x]) << shift);
    }
  }
}

// bl::ImageFilterInternal - Image Blur
// ====================================

static BLResult copy_image(BLImageCore* dst, const BLImageCore* src) noexcept {
  return bl_image_assign_deep(dst, src);
}

static BLResult blur_image(
  BLImageCore* dst, const BLImageCore* src,
  const BLBlurOptions* options, bool force_approximate, bool is_box_blur) noexcept {

  if (!options || !is_valid_blur_flags(options->flags))
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  BLImage local_src;
  const BLImageCore* src_core = src;

  if (dst == src) {
    BL_PROPAGATE(bl_image_assign_deep(&local_src, src));
    src_core = &local_src;
  }

  BLImageData src_data;
  BL_PROPAGATE(bl_image_get_data(src_core, &src_data));

  if (src_data.format == BL_FORMAT_NONE)
    return bl_image_reset(dst);

  if (!is_filter_format(BLFormat(src_data.format)))
    return bl_make_error(BL_ERROR_NOT_IMPLEMENTED);

  ResolvedBlur blur {};
  uint32_t passes = 1;

  if (is_box_blur)
    BL_PROPAGATE(resolve_box_blur(options, &blur, &passes));
  else
    BL_PROPAGATE(resolve_blur(options, &blur));

  if (blur.kernel_radius == 0)
    return copy_image(dst, src_core);

  bool approximate = force_approximate || (options->flags & BL_BLUR_FLAG_APPROXIMATE) != 0;
  EdgeMode edge_mode = edge_mode_from_flags(options->flags);

  int w = src_data.size.w;
  int h = src_data.size.h;
  size_t plane_size;

  BL_PROPAGATE(check_plane_size(w, h, &plane_size));

  BLArray<uint8_t> plane_array;
  uint8_t* plane;
  BL_PROPAGATE(alloc_array(plane_array, plane_size, &plane));

  BL_PROPAGATE(bl_image_create(dst, w, h, BLFormat(src_data.format)));

  BLImageData dst_data;
  BL_PROPAGATE(bl_image_make_mutable(dst, &dst_data));

  uint32_t channel_count = src_data.format == BL_FORMAT_A8 ? 1u : 4u;
  int radius = int(blur.radius);
  double sigma = blur.sigma;

  for (uint32_t channel = 0; channel < channel_count; channel++) {
    extract_plane(plane, static_cast<const uint8_t*>(src_data.pixel_data), src_data.stride, w, h, src_data.format, channel);

    if (is_box_blur)
      BL_PROPAGATE(box_blur_plane_multi(plane, intptr_t(w), plane, intptr_t(w), w, h, radius, passes, edge_mode));
    else
      BL_PROPAGATE(gaussian_blur_plane(plane, intptr_t(w), plane, intptr_t(w), w, h, blur.radius, sigma, edge_mode, approximate));

    store_plane(static_cast<uint8_t*>(dst_data.pixel_data), dst_data.stride, plane, w, h, dst_data.format, channel);
  }

  return BL_SUCCESS;
}

// bl::ImageFilterInternal - Shadow Helpers
// ========================================

static uint32_t source_alpha(const uint8_t* row, int x, uint32_t format) noexcept {
  if (format == BL_FORMAT_A8)
    return row[x];
  if (format == BL_FORMAT_XRGB32)
    return 255u;
  return (reinterpret_cast<const uint32_t*>(row)[x] >> 24) & 0xFFu;
}

static void shift_coverage(
  uint8_t* dst_data,
  const uint8_t* src_data,
  int w, int h, int dx, int dy, uint8_t outside_value) noexcept {

  for (int y = 0; y < h; y++) {
    int64_t sy = int64_t(y) - int64_t(dy);
    uint8_t* dst_row = dst_data + size_t(y) * size_t(w);

    if (sy < 0 || sy >= h) {
      memset(dst_row, outside_value, size_t(w));
      continue;
    }

    const uint8_t* src_row = src_data + size_t(sy) * size_t(w);

    if (dx == 0) {
      memcpy(dst_row, src_row, size_t(w));
    }
    else {
      for (int x = 0; x < w; x++) {
        int64_t sx = int64_t(x) - int64_t(dx);
        dst_row[x] = (sx >= 0 && sx < w) ? src_row[size_t(sx)] : outside_value;
      }
    }
  }

}

static BLResult blur_coverage(
  uint8_t* dst_data, const uint8_t* src_data,
  int w, int h, double radius, double sigma, uint32_t flags, EdgeMode edge_mode) noexcept {

  if (radius <= 0.0 && sigma <= 0.0)
    return copy_plane(dst_data, intptr_t(w), src_data, intptr_t(w), w, h);

  ResolvedBlur blur {};

  BLBlurOptions options {};
  options.reset(radius, sigma);
  BL_PROPAGATE(resolve_blur(&options, &blur));

  if (blur.kernel_radius == 0)
    return copy_plane(dst_data, intptr_t(w), src_data, intptr_t(w), w, h);

  return gaussian_blur_plane(dst_data, intptr_t(w), src_data, intptr_t(w), w, h, blur.radius, blur.sigma, edge_mode, (flags & BL_SHADOW_FLAG_APPROXIMATE) != 0);
}

static BL_INLINE uint32_t premultiply_shadow(uint32_t channel_value, uint32_t shadow_alpha) noexcept {
  return (channel_value * shadow_alpha + 127u) / 255u;
}

static void compose_pixel(
  uint8_t* dst, uint8_t src_r, uint8_t src_g, uint8_t src_b, uint8_t src_a,
  uint8_t shadow_r, uint8_t shadow_g, uint8_t shadow_b, uint8_t shadow_a,
  bool inner) noexcept {

  if (inner) {
    uint32_t factor = 255u - shadow_a;
    uint32_t r = (uint32_t(src_r) * factor + 127u) / 255u + shadow_r;
    uint32_t g = (uint32_t(src_g) * factor + 127u) / 255u + shadow_g;
    uint32_t b = (uint32_t(src_b) * factor + 127u) / 255u + shadow_b;
    uint32_t a = src_a;

    if (r > a) r = a;
    if (g > a) g = a;
    if (b > a) b = a;

    dst[0] = uint8_t(b);
    dst[1] = uint8_t(g);
    dst[2] = uint8_t(r);
    dst[3] = uint8_t(a);
  }
  else {
    uint32_t factor = 255u - src_a;
    uint32_t r = uint32_t(src_r) + (uint32_t(shadow_r) * factor + 127u) / 255u;
    uint32_t g = uint32_t(src_g) + (uint32_t(shadow_g) * factor + 127u) / 255u;
    uint32_t b = uint32_t(src_b) + (uint32_t(shadow_b) * factor + 127u) / 255u;
    uint32_t a = uint32_t(src_a) + (uint32_t(shadow_a) * factor + 127u) / 255u;

    if (r > a) r = a;
    if (g > a) g = a;
    if (b > a) b = a;

    dst[0] = uint8_t(b);
    dst[1] = uint8_t(g);
    dst[2] = uint8_t(r);
    dst[3] = uint8_t(a);
  }

}

// bl::ImageFilterInternal - Shadow
// ================================

static BLResult shadow_image(BLImageCore* dst, const BLImageCore* src, const BLShadowOptions* options) noexcept {
  if (!options || !is_valid_shadow_flags(options->flags))
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  if (!std::isfinite(options->offset.x) || !std::isfinite(options->offset.y) ||
      !std::isfinite(options->blur_radius) || !std::isfinite(options->blur_sigma))
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  if (std::fabs(options->offset.x) > 2147483647.0 || std::fabs(options->offset.y) > 2147483647.0)
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  if (options->blur_radius < 0.0 || options->blur_sigma < 0.0)
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  if (options->type != BL_SHADOW_TYPE_DROP && options->type != BL_SHADOW_TYPE_INNER)
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  BLImage local_src;
  const BLImageCore* src_core = src;

  if (dst == src) {
    BL_PROPAGATE(bl_image_assign_deep(&local_src, src));
    src_core = &local_src;
  }

  BLImageData src_data;
  BL_PROPAGATE(bl_image_get_data(src_core, &src_data));

  if (src_data.format == BL_FORMAT_NONE)
    return bl_image_reset(dst);

  if (!is_filter_format(BLFormat(src_data.format)))
    return bl_make_error(BL_ERROR_NOT_IMPLEMENTED);

  int w = src_data.size.w;
  int h = src_data.size.h;
  size_t plane_size;

  BL_PROPAGATE(check_plane_size(w, h, &plane_size));

  BLArray<uint8_t> work_array;
  BLArray<uint8_t> blur_array;
  uint8_t* work_data;
  uint8_t* blur_data;

  BL_PROPAGATE(alloc_array(work_array, plane_size, &work_data));
  BL_PROPAGATE(alloc_array(blur_array, plane_size, &blur_data));

  bool inner = options->type == BL_SHADOW_TYPE_INNER;
  bool shadow_only = (options->flags & BL_SHADOW_FLAG_SHADOW_ONLY) != 0;

  int dx = int(std::lround(options->offset.x));
  int dy = int(std::lround(options->offset.y));

  // Work contains the mask that is shifted. For a drop shadow it's the source alpha; for an inner shadow it's
  // the inverted source alpha (the area outside of the shape).
  for (int y = 0; y < h; y++) {
    const uint8_t* src_row = static_cast<const uint8_t*>(src_data.pixel_data) + intptr_t(y) * src_data.stride;
    uint8_t* work_row = work_data + size_t(y) * size_t(w);

    if (inner) {
      for (int x = 0; x < w; x++)
        work_row[x] = uint8_t(255u - source_alpha(src_row, x, src_data.format));
    }
    else {
      for (int x = 0; x < w; x++)
        work_row[x] = uint8_t(source_alpha(src_row, x, src_data.format));
    }
  }

  // Shift the mask by the integer offset. For an inner shadow the area outside of the image is considered
  // fully outside of the shape.
  shift_coverage(blur_data, work_data, w, h, dx, dy, inner ? 255u : 0u);

  // For a drop shadow the shifted alpha is blurred with transparent edges. For an inner shadow the inverted
  // alpha is blurred while the area outside of the image remains opaque, otherwise the shadow would be cut
  // at the image border.
  if (options->blur_radius > 0.0 || options->blur_sigma > 0.0) {
    BL_PROPAGATE(blur_coverage(work_data, blur_data, w, h, options->blur_radius, options->blur_sigma, options->flags, inner ? EdgeMode::kOpaque : EdgeMode::kZero));
    std::swap(work_data, blur_data);
  }
  else {
    std::swap(work_data, blur_data);
  }

  // `work_data` now contains shadow coverage.
  BL_PROPAGATE(bl_image_create(dst, w, h, BLFormat(src_data.format)));

  BLImageData dst_data;
  BL_PROPAGATE(bl_image_make_mutable(dst, &dst_data));

  uint32_t color_value = options->color.value;
  uint32_t color_a = (color_value >> 24) & 0xFFu;
  uint32_t color_r = (color_value >> 16) & 0xFFu;
  uint32_t color_g = (color_value >> 8) & 0xFFu;
  uint32_t color_b = color_value & 0xFFu;

  for (int y = 0; y < h; y++) {
    const uint8_t* src_row = static_cast<const uint8_t*>(src_data.pixel_data) + intptr_t(y) * src_data.stride;
    const uint8_t* coverage_row = work_data + size_t(y) * size_t(w);
    uint8_t* dst_row = static_cast<uint8_t*>(dst_data.pixel_data) + intptr_t(y) * dst_data.stride;

    if (src_data.format == BL_FORMAT_A8) {
      for (int x = 0; x < w; x++) {
        uint32_t src_a = source_alpha(src_row, x, src_data.format);
        uint32_t coverage = coverage_row[x];

        if (inner) {
        uint32_t shadow_a = (color_a * ((src_a * coverage + 127u) / 255u) + 127u) / 255u;
          dst_row[x] = uint8_t(shadow_only ? shadow_a : src_a);
        }
        else {
          uint32_t shadow_a = (color_a * coverage + 127u) / 255u;
          uint32_t out_a = shadow_only
            ? shadow_a
            : src_a + (shadow_a * (255u - src_a) + 127u) / 255u;
          dst_row[x] = uint8_t(out_a);
        }
      }
      continue;
    }

    for (int x = 0; x < w; x++) {
      uint32_t src_pixel = reinterpret_cast<const uint32_t*>(src_row)[x];
      uint32_t src_a = source_alpha(src_row, x, src_data.format);
      uint32_t coverage = coverage_row[x];

      uint32_t mask_a = inner ? (src_a * coverage + 127u) / 255u : coverage;
      uint32_t shadow_a = (color_a * mask_a + 127u) / 255u;
      uint8_t shadow_r = uint8_t(premultiply_shadow(color_r, shadow_a));
      uint8_t shadow_g = uint8_t(premultiply_shadow(color_g, shadow_a));
      uint8_t shadow_b = uint8_t(premultiply_shadow(color_b, shadow_a));

      if (shadow_only) {
        uint8_t* dst_pixel = dst_row + size_t(x) * 4u;
        dst_pixel[0] = shadow_b;
        dst_pixel[1] = shadow_g;
        dst_pixel[2] = shadow_r;
        dst_pixel[3] = src_data.format == BL_FORMAT_XRGB32 ? 0xFFu : uint8_t(shadow_a);
      }
      else {
        uint8_t* dst_pixel = dst_row + size_t(x) * 4u;
        uint8_t src_r = uint8_t((src_pixel >> 16) & 0xFFu);
        uint8_t src_g = uint8_t((src_pixel >> 8) & 0xFFu);
        uint8_t src_b = uint8_t(src_pixel & 0xFFu);

        compose_pixel(dst_pixel, src_r, src_g, src_b, uint8_t(src_a), shadow_r, shadow_g, shadow_b, uint8_t(shadow_a), inner);

        if (src_data.format == BL_FORMAT_XRGB32)
          dst_pixel[3] = 0xFF;
      }
    }
  }

  return BL_SUCCESS;
}

} // {ImageFilterInternal}
} // {bl}

// bl::ImageFilter - API - Gaussian Blur
// =====================================

BL_API_IMPL BLResult bl_image_gaussian_blur(BLImageCore* dst, const BLImageCore* src, const BLBlurOptions* options) noexcept {
  BL_ASSERT(dst->_d.is_image());
  BL_ASSERT(src->_d.is_image());

  return bl::ImageFilterInternal::blur_image(dst, src, options, false, false);
}

BL_API_IMPL BLResult bl_image_gaussian_blur_approx(BLImageCore* dst, const BLImageCore* src, const BLBlurOptions* options) noexcept {
  BL_ASSERT(dst->_d.is_image());
  BL_ASSERT(src->_d.is_image());

  return bl::ImageFilterInternal::blur_image(dst, src, options, true, false);
}

BL_API_IMPL BLResult bl_image_box_blur(BLImageCore* dst, const BLImageCore* src, const BLBlurOptions* options) noexcept {
  BL_ASSERT(dst->_d.is_image());
  BL_ASSERT(src->_d.is_image());

  return bl::ImageFilterInternal::blur_image(dst, src, options, false, true);
}

// bl::ImageFilter - API - Shadow
// ==============================

BL_API_IMPL BLResult bl_image_shadow(BLImageCore* dst, const BLImageCore* src, const BLShadowOptions* options) noexcept {
  BL_ASSERT(dst->_d.is_image());
  BL_ASSERT(src->_d.is_image());

  return bl::ImageFilterInternal::shadow_image(dst, src, options);
}

BL_API_IMPL BLResult bl_image_drop_shadow(
  BLImageCore* dst, const BLImageCore* src,
  const BLPoint* offset, double blur_radius, double blur_sigma,
  const BLRgba32* color, uint32_t flags) noexcept {

  BL_ASSERT(dst->_d.is_image());
  BL_ASSERT(src->_d.is_image());

  if (!offset || !color)
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  BLShadowOptions options {};
  options.offset = *offset;
  options.blur_radius = blur_radius;
  options.blur_sigma = blur_sigma;
  options.color = *color;
  options.type = BL_SHADOW_TYPE_DROP;
  options.flags = flags;

  return bl::ImageFilterInternal::shadow_image(dst, src, &options);
}

BL_API_IMPL BLResult bl_image_inner_shadow(
  BLImageCore* dst, const BLImageCore* src,
  const BLPoint* offset, double blur_radius, double blur_sigma,
  const BLRgba32* color, uint32_t flags) noexcept {

  BL_ASSERT(dst->_d.is_image());
  BL_ASSERT(src->_d.is_image());

  if (!offset || !color)
    return bl_make_error(BL_ERROR_INVALID_VALUE);

  BLShadowOptions options {};
  options.offset = *offset;
  options.blur_radius = blur_radius;
  options.blur_sigma = blur_sigma;
  options.color = *color;
  options.type = BL_SHADOW_TYPE_INNER;
  options.flags = flags;

  return bl::ImageFilterInternal::shadow_image(dst, src, &options);
}

// bl::BLImage - API - Filter Convenience Methods
// ==============================================

BLResult BLImage::gaussian_blur(BLImage& dst, const BLImage& src, double radius, double sigma) noexcept {
  return BLImageFilter::gaussian_blur(dst, src, radius, sigma);
}

BLResult BLImage::gaussian_blur_approx(BLImage& dst, const BLImage& src, double radius, double sigma) noexcept {
  return BLImageFilter::gaussian_blur_approx(dst, src, radius, sigma);
}

BLResult BLImage::box_blur(BLImage& dst, const BLImage& src, int radius, uint32_t passes) noexcept {
  return BLImageFilter::box_blur(dst, src, radius, passes);
}

BLResult BLImage::drop_shadow(
  BLImage& dst, const BLImage& src, const BLPoint& offset, double blur_radius, double blur_sigma,
  const BLRgba32& color, uint32_t flags) noexcept {
  return BLImageFilter::drop_shadow(dst, src, offset, blur_radius, blur_sigma, color, flags);
}

BLResult BLImage::inner_shadow(
  BLImage& dst, const BLImage& src, const BLPoint& offset, double blur_radius, double blur_sigma,
  const BLRgba32& color, uint32_t flags) noexcept {
  return BLImageFilter::inner_shadow(dst, src, offset, blur_radius, blur_sigma, color, flags);
}
