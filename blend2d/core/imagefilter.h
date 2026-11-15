// This file is part of Blend2D project <https://blend2d.com>
//
// See blend2d.h or LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

#ifndef BLEND2D_IMAGEFILTER_H_INCLUDED
#define BLEND2D_IMAGEFILTER_H_INCLUDED

#include <blend2d/core/geometry.h>
#include <blend2d/core/image.h>
#include <blend2d/core/rgba.h>

//! \addtogroup bl_imaging
//! \{

//! \name BLImageFilter - Constants
//! \{

//! Flags used by \ref BLBlurOptions.
BL_DEFINE_ENUM(BLBlurFlags) {
  //! No flags.
  BL_BLUR_NO_FLAGS = 0u,

  //! Pixels outside of the image are considered transparent. By default the image edges are clamped, which is
  //! the usual choice for background blur. Use this flag when blurring alpha masks or other effects that must
  //! not bleed from the image border.
  BL_BLUR_FLAG_TRANSPARENT_EDGE = 0x00000001u,

  //! Uses the fast three-pass box approximation instead of an exact Gaussian kernel. This flag is ignored by
  //! \ref BLImageFilter::box_blur().
  BL_BLUR_FLAG_APPROXIMATE = 0x00000002u

  BL_FORCE_ENUM_UINT32(BL_BLUR)
};

//! Shadow type used by \ref BLShadowOptions.
BL_DEFINE_ENUM(BLShadowType) {
  //! Drop shadow rendered behind the source.
  BL_SHADOW_TYPE_DROP = 0,
  //! Inner shadow rendered inside the source alpha.
  BL_SHADOW_TYPE_INNER = 1,

  //! Maximum value of \ref BLShadowType.
  BL_SHADOW_TYPE_MAX_VALUE = 1

  BL_FORCE_ENUM_UINT32(BL_SHADOW_TYPE)
};

//! Flags used by \ref BLShadowOptions.
BL_DEFINE_ENUM(BLShadowFlags) {
  //! No flags.
  BL_SHADOW_NO_FLAGS = 0u,

  //! Writes only the shadow layer and doesn't composite the source image into the destination. This is useful
  //! when the caller wants to blend the shadow with its own backdrop.
  BL_SHADOW_FLAG_SHADOW_ONLY = 0x00000001u,

  //! Uses the fast three-pass box approximation for the shadow blur.
  BL_SHADOW_FLAG_APPROXIMATE = 0x00000002u

  BL_FORCE_ENUM_UINT32(BL_SHADOW)
};

//! \}

//! \name BLImageFilter - Structs
//! \{

//! Parameters of \ref BLImageFilter::gaussian_blur() and \ref BLImageFilter::box_blur().
//!
//! `radius` specifies the maximum kernel extent in pixels. When `sigma` is zero it is derived as `radius / 3`.
//! When `radius` is zero it is derived as `ceil(3 * sigma)`. If both values are zero the filter is a no-op.
struct BLBlurOptions {
  //! Blur radius (kernel extent in pixels).
  double radius;
  //! Gaussian standard deviation. Zero means "derive from `radius`".
  double sigma;
  //! Number of box passes, used by \ref BLImageFilter::box_blur(). The default is one pass.
  uint32_t passes;
  //! Blur flags, see \ref BLBlurFlags.
  uint32_t flags;

#ifdef __cplusplus
  BL_INLINE_NODEBUG void reset() noexcept { *this = BLBlurOptions{}; }

  BL_INLINE_NODEBUG void reset(double radius_, double sigma_, uint32_t passes_ = 1, uint32_t flags_ = BL_BLUR_NO_FLAGS) noexcept {
    radius = radius_;
    sigma = sigma_;
    passes = passes_;
    flags = flags_;
  }
#endif
};

//! Parameters of \ref BLImageFilter::shadow().
struct BLShadowOptions {
  //! Shadow offset. Offsets are rounded to whole pixels.
  BLPoint offset;
  //! Shadow blur radius (kernel extent in pixels).
  double blur_radius;
  //! Shadow Gaussian standard deviation. Zero means "derive from `blur_radius`".
  double blur_sigma;
  //! Shadow color (RGBA).
  BLRgba32 color;
  //! Shadow type, see \ref BLShadowType.
  uint32_t type;
  //! Shadow flags, see \ref BLShadowFlags.
  uint32_t flags;

#ifdef __cplusplus
  BL_INLINE_NODEBUG void reset() noexcept { *this = BLShadowOptions{}; }
#endif
};

//! \}
//! \}

//! \addtogroup bl_c_api
//! \{

//! \name BLImageFilter - C API
//! \{

BL_BEGIN_C_DECLS

BL_API BLResult BL_CDECL bl_image_gaussian_blur(
  BLImageCore* dst, const BLImageCore* src, const BLBlurOptions* options) BL_NOEXCEPT_C;
BL_API BLResult BL_CDECL bl_image_gaussian_blur_approx(
  BLImageCore* dst, const BLImageCore* src, const BLBlurOptions* options) BL_NOEXCEPT_C;
BL_API BLResult BL_CDECL bl_image_box_blur(
  BLImageCore* dst, const BLImageCore* src, const BLBlurOptions* options) BL_NOEXCEPT_C;
BL_API BLResult BL_CDECL bl_image_shadow(
  BLImageCore* dst, const BLImageCore* src, const BLShadowOptions* options) BL_NOEXCEPT_C;
BL_API BLResult BL_CDECL bl_image_drop_shadow(
  BLImageCore* dst, const BLImageCore* src, const BLPoint* offset, double blur_radius, double blur_sigma,
  const BLRgba32* color, uint32_t flags) BL_NOEXCEPT_C;
BL_API BLResult BL_CDECL bl_image_inner_shadow(
  BLImageCore* dst, const BLImageCore* src, const BLPoint* offset, double blur_radius, double blur_sigma,
  const BLRgba32* color, uint32_t flags) BL_NOEXCEPT_C;

BL_END_C_DECLS

//! \}
//! \}

//! \addtogroup bl_imaging
//! \{

//! \name BLImageFilter - C++ API
//! \{

#ifdef __cplusplus
//! Image filter operations [C++ API].
//!
//! The filters are designed for the common UI effects: background blur, drop shadow, and inner shadow. All
//! operations write a new image to `dst` and accept `dst == src` (the source is copied internally before the
//! destination is re-created).
//!
//! Supported pixel formats are `BL_FORMAT_PRGB32`, `BL_FORMAT_XRGB32`, and `BL_FORMAT_A8`. Color channels of
//! premultiplied images are filtered as-is, which avoids dark halos around transparent edges.
class BLImageFilter final {
public:
  //! Applies an exact separable Gaussian blur to `src` and writes the result to `dst`.
  //!
  //! The image size and pixel format are preserved. `options->radius` limits the kernel extent while
  //! `options->sigma` controls the Gaussian distribution.
  static BL_INLINE_NODEBUG BLResult gaussian_blur(BLImage& dst, const BLImage& src, const BLBlurOptions& options) noexcept {
    return bl_image_gaussian_blur(&dst, &src, &options);
  }

  //! Applies a Gaussian blur using the three-pass box approximation.
  //!
  //! This is significantly faster for large radii and visually very close to an exact Gaussian. The
  //! approximation error decreases as `sigma` grows.
  static BL_INLINE_NODEBUG BLResult gaussian_blur_approx(BLImage& dst, const BLImage& src, const BLBlurOptions& options) noexcept {
    return bl_image_gaussian_blur_approx(&dst, &src, &options);
  }

  //! Applies a box blur to `src` and writes the result to `dst`.
  static BL_INLINE_NODEBUG BLResult box_blur(BLImage& dst, const BLImage& src, const BLBlurOptions& options) noexcept {
    return bl_image_box_blur(&dst, &src, &options);
  }

  //! Applies a blur described by `options`.
  static BL_INLINE_NODEBUG BLResult gaussian_blur(BLImage& dst, const BLImage& src, double radius, double sigma = 0.0) noexcept {
    BLBlurOptions options {};
    options.reset(radius, sigma);
    return bl_image_gaussian_blur(&dst, &src, &options);
  }

  //! Applies a fast box-approximated Gaussian blur described by `radius` and `sigma`.
  static BL_INLINE_NODEBUG BLResult gaussian_blur_approx(BLImage& dst, const BLImage& src, double radius, double sigma = 0.0) noexcept {
    BLBlurOptions options {};
    options.reset(radius, sigma);
    return bl_image_gaussian_blur_approx(&dst, &src, &options);
  }

  //! Applies a box blur with the given `radius` and number of `passes`.
  static BL_INLINE_NODEBUG BLResult box_blur(BLImage& dst, const BLImage& src, int radius, uint32_t passes = 1) noexcept {
    BLBlurOptions options {};
    options.reset(radius, 0.0, passes);
    return bl_image_box_blur(&dst, &src, &options);
  }

  //! Applies a drop or inner shadow described by `options`.
  static BL_INLINE_NODEBUG BLResult shadow(BLImage& dst, const BLImage& src, const BLShadowOptions& options) noexcept {
    return bl_image_shadow(&dst, &src, &options);
  }

  //! Applies a drop shadow and composites it behind `src`.
  static BL_INLINE_NODEBUG BLResult drop_shadow(
    BLImage& dst, const BLImage& src, const BLPoint& offset, double blur_radius, double blur_sigma,
    const BLRgba32& color, uint32_t flags = BL_SHADOW_NO_FLAGS) noexcept {
    return bl_image_drop_shadow(&dst, &src, &offset, blur_radius, blur_sigma, &color, flags);
  }

  //! Applies an inner shadow and composites it inside the alpha of `src`.
  static BL_INLINE_NODEBUG BLResult inner_shadow(
    BLImage& dst, const BLImage& src, const BLPoint& offset, double blur_radius, double blur_sigma,
    const BLRgba32& color, uint32_t flags = BL_SHADOW_NO_FLAGS) noexcept {
    return bl_image_inner_shadow(&dst, &src, &offset, blur_radius, blur_sigma, &color, flags);
  }
};
#endif

//! \}
//! \}

#endif // BLEND2D_IMAGEFILTER_H_INCLUDED
