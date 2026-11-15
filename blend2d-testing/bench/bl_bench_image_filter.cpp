#include <blend2d/blend2d.h>

#include <blend2d-testing/commons/cmdline.h>
#include <blend2d-testing/commons/performance_timer.h>

#include <cmath>
#include <cstdio>

template<typename T>
static void consume(T value) {
  volatile T stack_value = value;
  (void)stack_value;
}

static BLImage create_test_image(uint32_t w, uint32_t h) noexcept {
  BLImage image;
  if (image.create(int(w), int(h), BL_FORMAT_PRGB32) != BL_SUCCESS)
    return BLImage();

  BLImageData data;
  if (image.make_mutable(&data) != BL_SUCCESS)
    return BLImage();

  for (uint32_t y = 0; y < h; y++) {
    uint32_t* row = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(data.pixel_data) + intptr_t(y) * data.stride);

    for (uint32_t x = 0; x < w; x++) {
      uint32_t a = 64u + ((x * 3u + y * 5u) & 0x7Fu);
      uint32_t r = (x * 7u + y * 3u) & a;
      uint32_t g = (x * 3u + y * 7u) & a;
      uint32_t b = (x * 5u + y * 11u) & a;
      row[x] = (a << 24) | (r << 16) | (g << 8) | b;
    }
  }

  return image;
}

static void print_result(const char* name, const PerformanceTimer& timer, uint32_t repeat, uint32_t w, uint32_t h) noexcept {
  double total_ms = timer.duration();
  double per_iter_ms = total_ms / double(repeat);
  double megapixels = double(uint64_t(w) * uint64_t(h)) / 1000000.0;
  double throughput = megapixels * double(repeat) / (total_ms * 0.001);

  printf("  %-24s %8.3f ms  %7.2f ms/iter  %8.2f MP/s\n", name, total_ms, per_iter_ms, throughput);
}

int main(int argc, char* argv[]) {
  CmdLine cmd_line(argc, argv);

  uint32_t width = cmd_line.value_as_uint("--width", 512);
  uint32_t height = cmd_line.value_as_uint("--height", 512);
  uint32_t repeat = cmd_line.value_as_uint("--repeat", 10);
  uint32_t radius = cmd_line.value_as_uint("--radius", 16);
  double sigma = atof(cmd_line.value_of("--sigma", "5.0"));

  if (width == 0 || height == 0 || repeat == 0) {
    printf("Invalid image size or repeat count.\n");
    return 1;
  }

  BLImage src = create_test_image(width, height);
  if (src.is_empty()) {
    printf("Failed to create the test image.\n");
    return 1;
  }

  BLImage dst;
  printf("Image filters: %ux%u, radius=%u, sigma=%0.2f, repeat=%u\n\n", width, height, radius, sigma, repeat);

  PerformanceTimer timer;

  BLBlurOptions blur_options {};
  blur_options.reset(double(radius), sigma);

  BLBlurOptions box_options {};
  box_options.reset(double(radius), 0.0, 3);

  BLShadowOptions shadow_options {};
  shadow_options.offset = BLPoint(4.0, 4.0);
  shadow_options.blur_radius = double(radius);
  shadow_options.blur_sigma = sigma;
  shadow_options.color = BLRgba32(0, 0, 0, 128);
  shadow_options.type = BL_SHADOW_TYPE_DROP;

  // Warm up.
  BLImageFilter::gaussian_blur(dst, src, blur_options);

  timer.start();
  for (uint32_t i = 0; i < repeat; i++)
    consume(BLImageFilter::gaussian_blur(dst, src, blur_options));
  timer.stop();
  print_result("gaussian (exact)", timer, repeat, width, height);
  double exact_ms = timer.duration();

  timer.start();
  for (uint32_t i = 0; i < repeat; i++)
    consume(BLImageFilter::gaussian_blur_approx(dst, src, blur_options));
  timer.stop();
  print_result("gaussian (3x box)", timer, repeat, width, height);
  double approx_ms = timer.duration();

  timer.start();
  for (uint32_t i = 0; i < repeat; i++)
    consume(BLImageFilter::box_blur(dst, src, box_options));
  timer.stop();
  print_result("box blur (3 passes)", timer, repeat, width, height);

  timer.start();
  for (uint32_t i = 0; i < repeat; i++)
    consume(BLImageFilter::shadow(dst, src, shadow_options));
  timer.stop();
  print_result("drop shadow", timer, repeat, width, height);

  shadow_options.flags = BL_SHADOW_FLAG_APPROXIMATE;
  timer.start();
  for (uint32_t i = 0; i < repeat; i++)
    consume(BLImageFilter::shadow(dst, src, shadow_options));
  timer.stop();
  print_result("drop shadow (3x box)", timer, repeat, width, height);

  shadow_options.flags = BL_SHADOW_NO_FLAGS;
  shadow_options.type = BL_SHADOW_TYPE_INNER;
  timer.start();
  for (uint32_t i = 0; i < repeat; i++)
    consume(BLImageFilter::shadow(dst, src, shadow_options));
  timer.stop();
  print_result("inner shadow", timer, repeat, width, height);

  if (approx_ms > 0.0)
    printf("\nApproximate Gaussian speedup: %0.2fx\n", exact_ms / approx_ms);

  return 0;
}
