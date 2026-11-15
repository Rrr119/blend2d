// This file is part of Blend2D project <https://blend2d.com>
//
// See blend2d.h or LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

#include <blend2d/core/api-build_p.h>
#include <blend2d/core/imagefilter_p.h>

namespace bl {
namespace ImageFilterInternal {

// bl::ImageFilterInternal - Scalar Helpers
// ========================================

static BL_INLINE uint32_t edge_value(EdgeMode edge_mode) noexcept {
  return edge_mode == EdgeMode::kOpaque ? 255u : 0u;
}

static BL_INLINE uint32_t sample_horizontal(const uint8_t* row, int w, int x, EdgeMode edge_mode) noexcept {
  if (uint32_t(x) < uint32_t(w))
    return row[x];

  if (edge_mode == EdgeMode::kClamp)
    return row[x < 0 ? 0 : w - 1];

  return edge_value(edge_mode);
}

static BL_INLINE uint32_t sample_vertical(const uint8_t* data, intptr_t stride, int h, int y, int x, EdgeMode edge_mode) noexcept {
  if (uint32_t(y) < uint32_t(h))
    return data[intptr_t(y) * stride + x];

  if (edge_mode == EdgeMode::kClamp)
    return data[intptr_t(y < 0 ? 0 : h - 1) * stride + x];

  return edge_value(edge_mode);
}

static BL_INLINE uint8_t normalize_u8(uint32_t sum, uint32_t window) noexcept {
  return uint8_t((sum + window / 2u) / window);
}

static BL_INLINE uint8_t normalize_f32(float value) noexcept {
  if (value <= 0.0f)
    return 0;
  if (value >= 255.0f)
    return 255;
  return uint8_t(value + 0.5f);
}

template<typename T>
static BL_INLINE BLResult alloc_array(BLArray<T>& array, size_t size, T** data_out) noexcept {
  BL_PROPAGATE(array.resize(size, T(0)));
  return array.make_mutable(data_out);
}

// bl::ImageFilterInternal - Scalar Box Blur
// =========================================

BL_HIDDEN BLResult box_blur_plane_scalar(
  uint8_t* dst_data, intptr_t dst_stride,
  const uint8_t* src_data, intptr_t src_stride,
  int w, int h, int radius, EdgeMode edge_mode) noexcept {

  if (radius <= 0) {
    for (int y = 0; y < h; y++)
      memcpy(dst_data + intptr_t(y) * dst_stride, src_data + intptr_t(y) * src_stride, size_t(w));
    return BL_SUCCESS;
  }

  size_t plane_size = size_t(w) * size_t(h);
  BLArray<uint8_t> tmp_array;
  BLArray<uint32_t> sum_array;

  uint8_t* tmp_data;
  uint32_t* sums;

  BL_PROPAGATE(alloc_array(tmp_array, plane_size, &tmp_data));
  BL_PROPAGATE(alloc_array(sum_array, size_t(w), &sums));

  uint32_t window = uint32_t(radius) * 2u + 1u;

  // Horizontal pass.
  for (int y = 0; y < h; y++) {
    const uint8_t* src_row = src_data + intptr_t(y) * src_stride;
    uint8_t* tmp_row = tmp_data + size_t(y) * size_t(w);
    uint32_t sum = 0;

    for (int x = -radius; x <= radius; x++)
      sum += sample_horizontal(src_row, w, x, edge_mode);

    for (int x = 0; x < w; x++) {
      tmp_row[x] = normalize_u8(sum, window);
      sum += sample_horizontal(src_row, w, x + radius + 1, edge_mode);
      sum -= sample_horizontal(src_row, w, x - radius, edge_mode);
    }
  }

  // Vertical pass.
  for (int x = 0; x < w; x++) {
    uint32_t sum = 0;
    for (int y = -radius; y <= radius; y++)
      sum += sample_vertical(tmp_data, intptr_t(w), h, y, x, edge_mode);
    sums[x] = sum;
  }

  for (int y = 0; y < h; y++) {
    uint8_t* dst_row = dst_data + intptr_t(y) * dst_stride;

    for (int x = 0; x < w; x++)
      dst_row[x] = normalize_u8(sums[x], window);

    if (y + 1 < h) {
      int add_y = y + radius + 1;
      int sub_y = y - radius;

      for (int x = 0; x < w; x++) {
        uint32_t add_value = sample_vertical(tmp_data, intptr_t(w), h, add_y, x, edge_mode);
        uint32_t sub_value = sample_vertical(tmp_data, intptr_t(w), h, sub_y, x, edge_mode);
        sums[x] = sums[x] + add_value - sub_value;
      }
    }
  }

  return BL_SUCCESS;
}

// bl::ImageFilterInternal - Scalar Gaussian Blur
// ==============================================

BL_HIDDEN BLResult gaussian_blur_plane_scalar(
  uint8_t* dst_data, intptr_t dst_stride,
  const uint8_t* src_data, intptr_t src_stride,
  int w, int h, const float* kernel, int kernel_radius, EdgeMode edge_mode) noexcept {

  if (kernel_radius <= 0) {
    for (int y = 0; y < h; y++)
      memcpy(dst_data + intptr_t(y) * dst_stride, src_data + intptr_t(y) * src_stride, size_t(w));
    return BL_SUCCESS;
  }

  size_t plane_size = size_t(w) * size_t(h);
  BLArray<float> tmp_array;
  float* tmp_data;

  BL_PROPAGATE(alloc_array(tmp_array, plane_size, &tmp_data));

  // Horizontal pass.
  for (int y = 0; y < h; y++) {
    const uint8_t* src_row = src_data + intptr_t(y) * src_stride;
    float* tmp_row = tmp_data + size_t(y) * size_t(w);

    for (int x = 0; x < w; x++) {
      float sum = 0.0f;

      for (int k = -kernel_radius; k <= kernel_radius; k++)
        sum += kernel[k + kernel_radius] * float(sample_horizontal(src_row, w, x + k, edge_mode));

      tmp_row[x] = sum;
    }
  }

  // Vertical pass.
  for (int y = 0; y < h; y++) {
    uint8_t* dst_row = dst_data + intptr_t(y) * dst_stride;

    for (int x = 0; x < w; x++) {
      float sum = 0.0f;

      for (int k = -kernel_radius; k <= kernel_radius; k++) {
        int sample_y = y + k;
        float value;

        if (uint32_t(sample_y) < uint32_t(h))
          value = tmp_data[size_t(sample_y) * size_t(w) + size_t(x)];
        else if (edge_mode == EdgeMode::kClamp)
          value = tmp_data[size_t(sample_y < 0 ? 0 : h - 1) * size_t(w) + size_t(x)];
        else
          value = edge_mode == EdgeMode::kOpaque ? 255.0f : 0.0f;

        sum += kernel[k + kernel_radius] * value;
      }

      dst_row[x] = normalize_f32(sum);
    }
  }

  return BL_SUCCESS;
}

} // {ImageFilterInternal}
} // {bl}
