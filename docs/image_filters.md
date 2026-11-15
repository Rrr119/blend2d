# Image Filters

Blend2D provides blur and shadow filters through `BLImageFilter`. The implementation is CPU-based and operates
directly on `BLImage` pixel data, so it can be used for UI background blur, card shadows, and text inner shadows.

Supported pixel formats:

- `BL_FORMAT_PRGB32`
- `BL_FORMAT_XRGB32`
- `BL_FORMAT_A8`

Premultiplied channels are filtered as-is. This is the correct choice for `BL_FORMAT_PRGB32`, because filtering
straight (non-premultiplied) color channels creates dark halos around transparent edges.

## Gaussian Blur

`BLImageFilter::gaussian_blur()` applies an exact separable Gaussian convolution. `radius` limits the kernel
extent, while `sigma` controls the distribution:

- If `sigma == 0`, it is derived as `radius / 3`.
- If `radius == 0`, it is derived as `ceil(3 * sigma)`.
- If both are zero, the operation is a no-op.

The default edge mode clamps pixels outside of the image. Set `BL_BLUR_FLAG_TRANSPARENT_EDGE` when the image
contains an alpha mask that should not bleed from the border.

```cpp
BLBlurOptions blur {};
blur.radius = 16.0;
blur.sigma = 5.0;

BLImage blurred;
BLImageFilter::gaussian_blur(blurred, background, blur);
```

## Box Approximation

An exact Gaussian becomes expensive for large radii. `BLImageFilter::gaussian_blur_approx()` uses three
separable box passes whose widths are computed from `sigma` to approximate the Gaussian. The result is visually
very close for typical UI radii and is substantially faster.

`BLImageFilter::box_blur()` exposes the box filter directly and accepts the number of passes. Box passes run
horizontally and vertically and use SSE2 or AVX2 implementations when available, with a scalar fallback for
large radii.

```cpp
BLBlurOptions fast_blur {};
fast_blur.radius = 16.0;
fast_blur.sigma = 5.0;

BLImage blurred;
BLImageFilter::gaussian_blur_approx(blurred, background, fast_blur);

// Three explicit box passes with radius 8.
BLImage box_blurred;
BLImageFilter::box_blur(box_blurred, background, 8, 3);
```

## Shadows

Shadows are built from the source alpha. `BLImageFilter::shadow()` accepts `BLShadowOptions` and dispatches on
`BLShadowOptions::type`:

- `BL_SHADOW_TYPE_DROP` shifts and blurs the source alpha, then composites the shadow behind the source.
- `BL_SHADOW_TYPE_INNER` shifts and blurs the inverted source alpha, clips it to the source alpha, and composites
  it over the source.

Offsets are rounded to whole pixels. The destination has the same size as the source; content shifted outside of
the image is clipped. Shadow results use the source pixel format.

By default the source is included in the result. Set `BL_SHADOW_FLAG_SHADOW_ONLY` to get only the shadow layer.
This is useful when blending the shadow with a custom backdrop. Set `BL_SHADOW_FLAG_APPROXIMATE` to use the
three-pass box approximation for the shadow blur.

```cpp
BLShadowOptions shadow {};
shadow.offset = BLPoint(4.0, 4.0);
shadow.blur_radius = 12.0;
shadow.blur_sigma = 4.0;
shadow.color = BLRgba32(0, 0, 0, 128);
shadow.type = BL_SHADOW_TYPE_DROP;

BLImage shadowed;
BLImageFilter::shadow(shadowed, card, shadow);
```

Convenience overloads are also available:

```cpp
BLImage drop;
BLImageFilter::drop_shadow(drop, card, BLPoint(4.0, 4.0), 12.0, 4.0, BLRgba32(0, 0, 0, 128));

BLImage inner;
BLImageFilter::inner_shadow(inner, card, BLPoint(2.0, 2.0), 8.0, 3.0, BLRgba32(0, 0, 0, 128));
```

For `BL_FORMAT_A8`, shadows produce alpha coverage; color components are not available. Inner shadows on A8 keep
the source alpha when compositing the source, so use `BL_SHADOW_FLAG_SHADOW_ONLY` to obtain the coverage layer.

## C API

The C API mirrors the C++ API:

```c
BLBlurOptions options = { 16.0, 5.0, 1, BL_BLUR_NO_FLAGS };
bl_image_gaussian_blur(&dst, &src, &options);
bl_image_gaussian_blur_approx(&dst, &src, &options);

BLShadowOptions shadow = { { 4.0, 4.0 }, 12.0, 4.0, { 0x80000000u }, BL_SHADOW_TYPE_DROP, BL_SHADOW_NO_FLAGS };
bl_image_shadow(&dst, &src, &shadow);
```

## Performance

The repository includes a standalone benchmark:

```bash
cmake --build build/Release --target bl_bench_image_filter
build/Release/bl_bench_image_filter --width=512 --height=512 --radius=16 --sigma=5 --repeat=10
```

The benchmark reports exact Gaussian, three-pass approximated Gaussian, box blur, drop shadow, and inner shadow
throughput. On x86-64, the approximated Gaussian typically reaches an order of magnitude higher throughput than
the exact convolution at medium and large radii. The exact path remains available when maximum kernel fidelity is
required.
