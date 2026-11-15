// This file is part of Blend2D project <https://blend2d.com>
//
// See blend2d.h or LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

#ifndef BLEND2D_IMAGEFILTER_P_H_INCLUDED
#define BLEND2D_IMAGEFILTER_P_H_INCLUDED

#include <blend2d/core/api-internal_p.h>
#include <blend2d/core/imagefilter.h>

//! \cond INTERNAL
//! \addtogroup blend2d_internal
//! \{

namespace bl {
namespace ImageFilterInternal {

//! Edge handling used by the plane blur kernels.
enum class EdgeMode : uint32_t {
  kClamp = 0,
  kZero = 1,
  kOpaque = 2
};

//! Performs one separable box-blur pass (horizontal followed by vertical).
BL_HIDDEN BLResult box_blur_plane_scalar(
  uint8_t* dst_data, intptr_t dst_stride,
  const uint8_t* src_data, intptr_t src_stride,
  int w, int h, int radius, EdgeMode edge_mode) noexcept;

//! Performs one separable Gaussian pass using the given normalized kernel.
BL_HIDDEN BLResult gaussian_blur_plane_scalar(
  uint8_t* dst_data, intptr_t dst_stride,
  const uint8_t* src_data, intptr_t src_stride,
  int w, int h, const float* kernel, int kernel_radius, EdgeMode edge_mode) noexcept;

#ifdef BL_BUILD_OPT_SSE2
BL_HIDDEN BLResult box_blur_plane_sse2(
  uint8_t* dst_data, intptr_t dst_stride,
  const uint8_t* src_data, intptr_t src_stride,
  int w, int h, int radius, EdgeMode edge_mode) noexcept;
#endif

#ifdef BL_BUILD_OPT_AVX2
BL_HIDDEN BLResult box_blur_plane_avx2(
  uint8_t* dst_data, intptr_t dst_stride,
  const uint8_t* src_data, intptr_t src_stride,
  int w, int h, int radius, EdgeMode edge_mode) noexcept;
#endif

} // {ImageFilterInternal}
} // {bl}

//! \}
//! \endcond

#endif // BLEND2D_IMAGEFILTER_P_H_INCLUDED
