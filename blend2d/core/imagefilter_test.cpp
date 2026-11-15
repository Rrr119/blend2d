// This file is part of Blend2D project <https://blend2d.com>
//
// See blend2d.h or LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

#include <blend2d/core/api-build_test_p.h>
#if defined(BL_TEST)

#include <blend2d/core/imagefilter_p.h>
#include <blend2d/core/runtime_p.h>

// bl::ImageFilter - Tests
// =======================

namespace bl {
namespace Tests {

static BLImage make_image(int w, int h, BLFormat format) noexcept {
  BLImage image;
  EXPECT_SUCCESS(image.create(w, h, format));
  return image;
}

static uint8_t* image_row(const BLImageData& data, int y) noexcept {
  return static_cast<uint8_t*>(data.pixel_data) + intptr_t(y) * data.stride;
}

static uint32_t image_pixel(const BLImageData& data, int x, int y) noexcept {
  return reinterpret_cast<const uint32_t*>(image_row(data, y))[x];
}

static void fill_a8(BLImage& image, uint8_t value) noexcept {
  BLImageData data;
  EXPECT_SUCCESS(image.make_mutable(&data));

  for (int y = 0; y < data.size.h; y++)
    memset(image_row(data, y), value, size_t(data.size.w));
}

static void test_constant_blur() noexcept {
  INFO("Testing constant image preservation");

  BLImage src = make_image(32, 24, BL_FORMAT_PRGB32);
  BLImageData src_data;
  EXPECT_SUCCESS(src.make_mutable(&src_data));

  for (int y = 0; y < src_data.size.h; y++) {
    uint32_t* row = reinterpret_cast<uint32_t*>(image_row(src_data, y));
    for (int x = 0; x < src_data.size.w; x++)
      row[x] = 0x50302010u;
  }

  BLBlurOptions options {};
  options.reset(5.0, 2.0);

  BLImage dst;
  EXPECT_SUCCESS(BLImageFilter::gaussian_blur(dst, src, options));

  BLImageData dst_data;
  EXPECT_SUCCESS(dst.get_data(&dst_data));

  for (int y = 0; y < dst_data.size.h; y++) {
    const uint32_t* row = reinterpret_cast<const uint32_t*>(image_row(dst_data, y));
    for (int x = 0; x < dst_data.size.w; x++) {
      uint32_t pixel = row[x];
      EXPECT_TRUE(((pixel >> 24) & 0xFFu) >= 0x4Fu && ((pixel >> 24) & 0xFFu) <= 0x51u);
      EXPECT_TRUE(((pixel >> 16) & 0xFFu) >= 0x2Fu && ((pixel >> 16) & 0xFFu) <= 0x31u);
    }
  }

  EXPECT_SUCCESS(BLImageFilter::box_blur(dst, src, 4, 2));
  EXPECT_SUCCESS(dst.get_data(&dst_data));

  for (int y = 0; y < dst_data.size.h; y++) {
    const uint32_t* row = reinterpret_cast<const uint32_t*>(image_row(dst_data, y));
    for (int x = 0; x < dst_data.size.w; x++) {
      uint32_t pixel = row[x];
      EXPECT_TRUE(((pixel >> 24) & 0xFFu) >= 0x4Fu && ((pixel >> 24) & 0xFFu) <= 0x51u);
      EXPECT_TRUE(((pixel >> 16) & 0xFFu) >= 0x2Fu && ((pixel >> 16) & 0xFFu) <= 0x31u);
    }
  }
}

static void test_gaussian_impulse() noexcept {
  INFO("Testing Gaussian impulse response");

  BLImage src = make_image(31, 31, BL_FORMAT_A8);
  fill_a8(src, 0);

  BLImageData src_data;
  EXPECT_SUCCESS(src.make_mutable(&src_data));
  image_row(src_data, 15)[15] = 255;

  BLBlurOptions options {};
  options.reset(6.0, 2.0, 1, BL_BLUR_FLAG_TRANSPARENT_EDGE);

  BLImage exact;
  EXPECT_SUCCESS(BLImageFilter::gaussian_blur(exact, src, options));

  BLImage approx;
  EXPECT_SUCCESS(BLImageFilter::gaussian_blur_approx(approx, src, options));

  BLImageData exact_data;
  BLImageData approx_data;
  EXPECT_SUCCESS(exact.get_data(&exact_data));
  EXPECT_SUCCESS(approx.get_data(&approx_data));

  uint32_t exact_sum = 0;
  uint32_t approx_sum = 0;

  for (int y = 0; y < 31; y++) {
    const uint8_t* exact_row = image_row(exact_data, y);
    const uint8_t* approx_row = image_row(approx_data, y);

    for (int x = 0; x < 31; x++) {
      exact_sum += exact_row[x];
      approx_sum += approx_row[x];

      int mirror_x = 30 - x;
      int mirror_y = 30 - y;
      EXPECT_EQ(exact_row[x], image_row(exact_data, y)[mirror_x]);
      EXPECT_EQ(exact_row[x], image_row(exact_data, mirror_y)[x]);
    }
  }

  EXPECT_TRUE(exact_sum >= 220u && exact_sum <= 260u);
  EXPECT_TRUE(approx_sum >= 245u && approx_sum <= 265u);

  uint8_t exact_center = image_row(exact_data, 15)[15];
  uint8_t exact_neighbor = image_row(exact_data, 15)[16];
  EXPECT_TRUE(exact_center > exact_neighbor);
  EXPECT_TRUE(exact_neighbor > 0);
}

static void test_box_impulse() noexcept {
  INFO("Testing box blur impulse response");

  BLImage src = make_image(9, 9, BL_FORMAT_A8);
  fill_a8(src, 0);

  BLImageData src_data;
  EXPECT_SUCCESS(src.make_mutable(&src_data));
  image_row(src_data, 4)[4] = 255;

  BLBlurOptions options {};
  options.reset(1.0, 0.0, 1, BL_BLUR_FLAG_TRANSPARENT_EDGE);

  BLImage dst;
  EXPECT_SUCCESS(BLImageFilter::box_blur(dst, src, options));

  BLImageData dst_data;
  EXPECT_SUCCESS(dst.get_data(&dst_data));

  for (int y = 3; y <= 5; y++) {
    for (int x = 3; x <= 5; x++)
      EXPECT_EQ(image_row(dst_data, y)[x], 28u);
  }

  EXPECT_EQ(image_row(dst_data, 2)[4], 0u);
  EXPECT_EQ(image_row(dst_data, 6)[4], 0u);
}

static void test_simd_box_blur() noexcept {
  INFO("Testing SIMD box blur parity");

  constexpr int kW = 37;
  constexpr int kH = 29;

  BLArray<uint8_t> source;
  BLArray<uint8_t> reference;
  BLArray<uint8_t> optimized;

  EXPECT_SUCCESS(source.resize(size_t(kW) * size_t(kH), uint8_t(0)));

  uint8_t* source_data;
  EXPECT_SUCCESS(source.make_mutable(&source_data));

  for (int y = 0; y < kH; y++) {
    for (int x = 0; x < kW; x++)
      source_data[size_t(y) * kW + x] = uint8_t((x * 37 + y * 53 + (x ^ y) * 11) & 0xFF);
  }

  const size_t plane_size = size_t(kW) * size_t(kH);
  EXPECT_SUCCESS(reference.resize(plane_size, uint8_t(0)));
  EXPECT_SUCCESS(optimized.resize(plane_size, uint8_t(0)));

  uint8_t* ref_data;
  uint8_t* opt_data;
  EXPECT_SUCCESS(reference.make_mutable(&ref_data));
  EXPECT_SUCCESS(optimized.make_mutable(&opt_data));

  const int radii[] = { 1, 2, 5, 17 };
  const ImageFilterInternal::EdgeMode edge_modes[] = {
    ImageFilterInternal::EdgeMode::kClamp,
    ImageFilterInternal::EdgeMode::kZero,
    ImageFilterInternal::EdgeMode::kOpaque
  };

  for (int radius : radii) {
    for (ImageFilterInternal::EdgeMode edge_mode : edge_modes) {
      EXPECT_SUCCESS(ImageFilterInternal::box_blur_plane_scalar(ref_data, kW, source_data, kW, kW, kH, radius, edge_mode));

#if defined(BL_BUILD_OPT_SSE2)
      if (bl_runtime_has_sse2(&bl_runtime_context)) {
        EXPECT_SUCCESS(ImageFilterInternal::box_blur_plane_sse2(opt_data, kW, source_data, kW, kW, kH, radius, edge_mode));
        for (size_t i = 0; i < plane_size; i++)
          EXPECT_TRUE(ref_data[i] + 1u >= opt_data[i] && ref_data[i] <= opt_data[i] + 1u);
      }
#endif

#if defined(BL_BUILD_OPT_AVX2)
      if (bl_runtime_has_avx2(&bl_runtime_context)) {
        EXPECT_SUCCESS(ImageFilterInternal::box_blur_plane_avx2(opt_data, kW, source_data, kW, kW, kH, radius, edge_mode));
        for (size_t i = 0; i < plane_size; i++)
          EXPECT_TRUE(ref_data[i] + 1u >= opt_data[i] && ref_data[i] <= opt_data[i] + 1u);
      }
#endif
    }
  }
}

static void test_shadows() noexcept {
  INFO("Testing drop and inner shadows");

  BLImage src = make_image(16, 16, BL_FORMAT_PRGB32);
  BLImageData src_data;
  EXPECT_SUCCESS(src.make_mutable(&src_data));

  for (int y = 0; y < 16; y++)
    memset(image_row(src_data, y), 0, 16 * 4);

  for (int y = 4; y < 8; y++) {
    uint32_t* row = reinterpret_cast<uint32_t*>(image_row(src_data, y));
    for (int x = 4; x < 8; x++)
      row[x] = 0xFFFF0000u;
  }

  BLImage drop;
  EXPECT_SUCCESS(BLImageFilter::drop_shadow(drop, src, BLPoint(3.0, 0.0), 0.0, 0.0, BLRgba32(0, 0, 0, 128)));

  BLImageData drop_data;
  EXPECT_SUCCESS(drop.get_data(&drop_data));

  uint32_t source_pixel = image_pixel(drop_data, 5, 5);
  EXPECT_EQ((source_pixel >> 16) & 0xFFu, 255u);
  EXPECT_EQ((source_pixel >> 24) & 0xFFu, 255u);

  uint32_t shadow_pixel = image_pixel(drop_data, 9, 5);
  EXPECT_TRUE(((shadow_pixel >> 24) & 0xFFu) >= 127u && ((shadow_pixel >> 24) & 0xFFu) <= 129u);
  EXPECT_EQ(shadow_pixel & 0x00FFFFFFu, 0u);

  EXPECT_EQ((image_pixel(drop_data, 12, 5) >> 24) & 0xFFu, 0u);

  BLImage shadow_only;
  EXPECT_SUCCESS(BLImageFilter::drop_shadow(shadow_only, src, BLPoint(3.0, 0.0), 0.0, 0.0, BLRgba32(0, 0, 0, 128), BL_SHADOW_FLAG_SHADOW_ONLY));

  BLImageData shadow_only_data;
  EXPECT_SUCCESS(shadow_only.get_data(&shadow_only_data));
  EXPECT_EQ((image_pixel(shadow_only_data, 5, 5) >> 24) & 0xFFu, 0u);
  EXPECT_TRUE(((image_pixel(shadow_only_data, 9, 5) >> 24) & 0xFFu) >= 127u);

  BLImage inner;
  EXPECT_SUCCESS(BLImageFilter::inner_shadow(inner, src, BLPoint(1.0, 1.0), 0.0, 0.0, BLRgba32(0, 0, 0, 128)));

  BLImageData inner_data;
  EXPECT_SUCCESS(inner.get_data(&inner_data));

  uint32_t edge_pixel = image_pixel(inner_data, 4, 4);
  EXPECT_EQ((edge_pixel >> 24) & 0xFFu, 255u);
  EXPECT_TRUE(((edge_pixel >> 16) & 0xFFu) >= 126u && ((edge_pixel >> 16) & 0xFFu) <= 128u);

  uint32_t center_pixel = image_pixel(inner_data, 5, 5);
  EXPECT_EQ((center_pixel >> 16) & 0xFFu, 255u);
  EXPECT_EQ((image_pixel(inner_data, 0, 0) >> 24) & 0xFFu, 0u);
}

static void test_alias_and_validation() noexcept {
  INFO("Testing in-place filtering and invalid arguments");

  BLImage image = make_image(16, 16, BL_FORMAT_A8);
  fill_a8(image, 100);

  EXPECT_SUCCESS(BLImage::gaussian_blur(image, image, 3.0, 1.0));

  BLImageData data;
  EXPECT_SUCCESS(image.get_data(&data));
  EXPECT_TRUE(image_row(data, 8)[8] >= 99u && image_row(data, 8)[8] <= 101u);

  BLBlurOptions options {};
  options.reset(-1.0, 1.0);
  EXPECT_EQ(BLImageFilter::gaussian_blur(image, image, options), BL_ERROR_INVALID_VALUE);

  BLImage empty;
  EXPECT_SUCCESS(BLImageFilter::gaussian_blur(image, empty, 1.0, 1.0));
  EXPECT_TRUE(image.is_empty());
}

static void test_edge_cases() noexcept {
  INFO("Testing degenerate sizes, large radii, and XRGB32");

  BLImage one = make_image(1, 1, BL_FORMAT_A8);
  fill_a8(one, 200);

  BLImage dst;
  EXPECT_SUCCESS(BLImageFilter::gaussian_blur(dst, one, 4.0, 1.5));

  BLImageData dst_data;
  EXPECT_SUCCESS(dst.get_data(&dst_data));
  EXPECT_EQ(image_row(dst_data, 0)[0], 200u);

  BLImage line = make_image(23, 1, BL_FORMAT_PRGB32);
  BLImageData line_data;
  EXPECT_SUCCESS(line.make_mutable(&line_data));

  for (int x = 0; x < 23; x++)
    reinterpret_cast<uint32_t*>(image_row(line_data, 0))[x] = 0x80FFFFFFu;

  EXPECT_SUCCESS(BLImageFilter::box_blur(dst, line, 200, 3));
  EXPECT_SUCCESS(BLImageFilter::gaussian_blur_approx(dst, line, 10.0, 3.0));

  BLImage xrgb = make_image(17, 13, BL_FORMAT_XRGB32);
  BLImageData xrgb_data;
  EXPECT_SUCCESS(xrgb.make_mutable(&xrgb_data));

  for (int y = 0; y < 13; y++) {
    uint32_t* row = reinterpret_cast<uint32_t*>(image_row(xrgb_data, y));
    for (int x = 0; x < 17; x++)
      row[x] = uint32_t(x * 13 + y * 17) << 16;
  }

  EXPECT_SUCCESS(BLImageFilter::gaussian_blur(dst, xrgb, 3.0, 1.0));
  EXPECT_SUCCESS(dst.get_data(&dst_data));
  EXPECT_EQ(dst_data.format, BL_FORMAT_XRGB32);
  EXPECT_EQ((image_pixel(dst_data, 8, 6) >> 24) & 0xFFu, 255u);

  BLImage a8_shadow;
  EXPECT_SUCCESS(BLImageFilter::drop_shadow(a8_shadow, one, BLPoint(0.0, 0.0), 2.0, 1.0, BLRgba32(0, 0, 0, 128), BL_SHADOW_FLAG_SHADOW_ONLY));
  EXPECT_SUCCESS(a8_shadow.get_data(&dst_data));
  EXPECT_TRUE(image_row(dst_data, 0)[0] >= 99u && image_row(dst_data, 0)[0] <= 101u);
}

UNIT(image_filter, BL_TEST_GROUP_IMAGE_UTILITIES) {
  test_constant_blur();
  test_gaussian_impulse();
  test_box_impulse();
  test_simd_box_blur();
  test_shadows();
  test_alias_and_validation();
  test_edge_cases();
}

} // {Tests}
} // {bl}

#endif
