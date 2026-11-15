// This file is part of Blend2D project <https://blend2d.com>
//
// See blend2d.h or LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

#include <blend2d/core/api-build_p.h>
#if defined(BL_TARGET_OPT_AVX2)

#include <blend2d/core/imagefilter_p.h>
#include <blend2d/simd/simd_p.h>

namespace bl {
namespace ImageFilterInternal {

// bl::ImageFilterInternal - AVX2 Helpers
// ======================================

static BL_INLINE __m256i load_u8x16_as_u16(const uint8_t* p) noexcept {
  __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_cvtepu8_epi16(v);
}

static BL_INLINE void store_normalized_u16x8(__m128i acc, float scale, uint8_t* dst) noexcept {
  const __m128i zero = _mm_setzero_si128();
  __m128i lo = _mm_unpacklo_epi16(acc, zero);
  __m128i hi = _mm_unpackhi_epi16(acc, zero);
  __m128 scale_v = _mm_set1_ps(scale);

  lo = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(lo), scale_v));
  hi = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(hi), scale_v));

  __m128i packed16 = _mm_packus_epi32(lo, hi);
  __m128i packed8 = _mm_packus_epi16(packed16, packed16);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), packed8);
}

static BL_INLINE void store_normalized_u16x16(__m256i acc, float scale, uint8_t* dst) noexcept {
  store_normalized_u16x8(_mm256_castsi256_si128(acc), scale, dst);
  store_normalized_u16x8(_mm256_extracti128_si256(acc, 1), scale, dst + 8);
}

static BL_INLINE void store_normalized_u32x8(__m256i acc, float scale, uint8_t* dst) noexcept {
  __m256 scale_v = _mm256_set1_ps(scale);
  acc = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(acc), scale_v));

  __m128i lo = _mm256_castsi256_si128(acc);
  __m128i hi = _mm256_extracti128_si256(acc, 1);
  __m128i packed16 = _mm_packs_epi32(lo, hi);
  __m128i packed8 = _mm_packus_epi16(packed16, packed16);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), packed8);
}

static BL_INLINE const uint8_t* row_at(const uint8_t* data, intptr_t stride, int h, int y, EdgeMode edge_mode) noexcept {
  if (uint32_t(y) < uint32_t(h))
    return data + intptr_t(y) * stride;
  if (edge_mode == EdgeMode::kClamp)
    return data + intptr_t(y < 0 ? 0 : h - 1) * stride;
  return nullptr;
}

static BL_INLINE uint32_t sample_vertical_tail(const uint8_t* data, intptr_t stride, int h, int y, int x, EdgeMode edge_mode) noexcept {
  if (uint32_t(y) < uint32_t(h))
    return data[intptr_t(y) * stride + x];
  if (edge_mode == EdgeMode::kClamp)
    return data[intptr_t(y < 0 ? 0 : h - 1) * stride + x];
  return edge_mode == EdgeMode::kOpaque ? 255u : 0u;
}

static BL_INLINE void add_constant_255(__m256i& acc) noexcept {
  acc = _mm256_add_epi16(acc, _mm256_set1_epi16(0x00FF));
}

static BL_INLINE void sub_constant_255(__m256i& acc) noexcept {
  acc = _mm256_sub_epi16(acc, _mm256_set1_epi16(0x00FF));
}

// bl::ImageFilterInternal - AVX2 Box Blur
// =======================================

BL_HIDDEN BLResult box_blur_plane_avx2(
  uint8_t* dst_data, intptr_t dst_stride,
  const uint8_t* src_data, intptr_t src_stride,
  int w, int h, int radius, EdgeMode edge_mode) noexcept {

  BL_ASSERT(radius > 0);
  BL_ASSERT(radius <= 128);

  size_t plane_size = size_t(w) * size_t(h);
  size_t padded_size = size_t(w) + size_t(radius) * 2u;

  BLArray<uint8_t> tmp_array;
  BLArray<uint8_t> padded_array;
  BLArray<uint32_t> prefix_array;

  uint8_t* tmp_data;
  uint8_t* padded_row;
  uint32_t* prefix;

  BL_PROPAGATE(tmp_array.resize(plane_size, uint8_t(0)));
  BL_PROPAGATE(padded_array.resize(padded_size, uint8_t(0)));
  BL_PROPAGATE(prefix_array.resize(padded_size + 1u, uint32_t(0)));
  BL_PROPAGATE(tmp_array.make_mutable(&tmp_data));
  BL_PROPAGATE(padded_array.make_mutable(&padded_row));
  BL_PROPAGATE(prefix_array.make_mutable(&prefix));

  int window = radius * 2 + 1;
  float scale = 1.0f / float(window);

  // Horizontal pass.
  for (int y = 0; y < h; y++) {
    const uint8_t* src_row = src_data + intptr_t(y) * src_stride;
    uint8_t* tmp_row = tmp_data + size_t(y) * size_t(w);

    uint8_t fill_left = edge_mode == EdgeMode::kClamp ? src_row[0] : (edge_mode == EdgeMode::kOpaque ? 0xFFu : 0u);
    uint8_t fill_right = edge_mode == EdgeMode::kClamp ? src_row[w - 1] : (edge_mode == EdgeMode::kOpaque ? 0xFFu : 0u);

    memset(padded_row, fill_left, size_t(radius));
    memcpy(padded_row + radius, src_row, size_t(w));
    memset(padded_row + radius + w, fill_right, size_t(radius));

    prefix[0] = 0;
    for (int i = 0; i < int(padded_size); i++)
      prefix[i + 1] = prefix[i] + padded_row[i];

    int x = 0;
    for (; x + 8 <= w; x += 8) {
      __m256i sum = _mm256_sub_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prefix + x + window)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prefix + x)));

      store_normalized_u32x8(sum, scale, tmp_row + x);
    }

    for (; x < w; x++) {
      uint32_t sum = prefix[x + window] - prefix[x];
      tmp_row[x] = uint8_t((sum + uint32_t(window / 2)) / uint32_t(window));
    }
  }

  // Vertical pass.
  int x = 0;

  for (; x + 16 <= w; x += 16) {
    uint16_t acc_values[16];

    for (int k = 0; k < 16; k++) {
      uint32_t sum = 0;

      for (int y = -radius; y <= radius; y++) {
        const uint8_t* row = row_at(tmp_data, intptr_t(w), h, y, edge_mode);
        sum += row ? row[x + k] : (edge_mode == EdgeMode::kOpaque ? 255u : 0u);
      }

      acc_values[k] = uint16_t(sum);
    }

    __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc_values));

    for (int y = 0; y < h; y++) {
      store_normalized_u16x16(acc, scale, dst_data + intptr_t(y) * dst_stride + x);

      if (y + 1 == h)
        break;

      const uint8_t* add_row = row_at(tmp_data, intptr_t(w), h, y + radius + 1, edge_mode);
      const uint8_t* sub_row = row_at(tmp_data, intptr_t(w), h, y - radius, edge_mode);

      if (add_row)
        acc = _mm256_add_epi16(acc, load_u8x16_as_u16(add_row + x));
      else if (edge_mode == EdgeMode::kOpaque)
        add_constant_255(acc);

      if (sub_row)
        acc = _mm256_sub_epi16(acc, load_u8x16_as_u16(sub_row + x));
      else if (edge_mode == EdgeMode::kOpaque)
        sub_constant_255(acc);
    }
  }

  // Vertical tail.
  for (; x < w; x++) {
    uint32_t sum = 0;

    for (int y = -radius; y <= radius; y++)
      sum += sample_vertical_tail(tmp_data, intptr_t(w), h, y, x, edge_mode);

    for (int y = 0; y < h; y++) {
      dst_data[intptr_t(y) * dst_stride + x] = uint8_t((sum + uint32_t(window / 2)) / uint32_t(window));

      if (y + 1 == h)
        break;

      sum += sample_vertical_tail(tmp_data, intptr_t(w), h, y + radius + 1, x, edge_mode);
      sum -= sample_vertical_tail(tmp_data, intptr_t(w), h, y - radius, x, edge_mode);
    }
  }

  return BL_SUCCESS;
}

} // {ImageFilterInternal}
} // {bl}

#endif // BL_TARGET_OPT_AVX2
