# MIMG Image Processing Examples

This gallery showcases various image processing modifiers available in mimg.

All examples are generated from the original `lena.png` image.

*Special thanks to Lena Söderberg, whose iconic photograph has been a cornerstone of image processing research for decades.*

## Original Image

![Original](examples/lena.png)

## Individual Modifiers

### Invert

Invert all colors

![Invert](examples/gallery/invert_lena.png)

### Grayscale

Convert to grayscale

![Grayscale](examples/gallery/grayscale_lena.png)

### Brightness +50

Increase brightness

![Brightness +50](examples/gallery/brightness_50_lena.png)

### Brightness -30

Decrease brightness

![Brightness -30](examples/gallery/brightness_-30_lena.png)

### Contrast 1.5

Increase contrast

![Contrast 1.5](examples/gallery/contrast_1.5_lena.png)

### Contrast 0.7

Decrease contrast

![Contrast 0.7](examples/gallery/contrast_0.7_lena.png)

### Saturation 1.5

Increase saturation

![Saturation 1.5](examples/gallery/saturation_1.5_lena.png)

### Saturation 0.5

Decrease saturation

![Saturation 0.5](examples/gallery/saturation_0.5_lena.png)

### Gamma 2.2

Apply gamma correction (brighter)

![Gamma 2.2](examples/gallery/gamma_2.2_lena.png)

### Gamma 0.45

Apply gamma correction (darker)

![Gamma 0.45](examples/gallery/gamma_0.45_lena.png)

### Sepia

Apply sepia tone effect

![Sepia](examples/gallery/sepia_lena.png)

### Blur

Apply box blur with kernel size 3

![Blur](examples/gallery/blur_3_lena.png)

### Sharpen

Sharpen the image

![Sharpen](examples/gallery/sharpen_lena.png)

### Flip Horizontal

Mirror horizontally

![Flip Horizontal](examples/gallery/flip_horizontal_lena.png)

### Flip Vertical

Mirror vertically

![Flip Vertical](examples/gallery/flip_vertical_lena.png)

### Rotate 45°

Rotate 45 degrees clockwise

![Rotate 45°](examples/gallery/rotate_45_lena.png)

### Resize

Resize to 200x200

![Resize](examples/gallery/resize_200_200_lena.png)

### Crop

Crop to center 200x200

![Crop](examples/gallery/crop_25_25_200_200_lena.png)

## Modifier Combinations

### Grayscale + Sepia

Vintage effect

![Grayscale + Sepia](examples/gallery/grayscale_sepia_lena.png)

### Brightness + Contrast + Sharpen

Enhanced image

![Brightness + Contrast + Sharpen](examples/gallery/brightness_30_contrast_1.2_sharpen_lena.png)

### Saturation + Gamma

Color corrected

![Saturation + Gamma](examples/gallery/saturation_0.3_gamma_1.5_lena.png)

### Blur + Sharpen

Noise reduction with detail enhancement

![Blur + Sharpen](examples/gallery/blur_5_sharpen_lena.png)

### Sepia + Brightness

Warm vintage tone

![Sepia + Brightness](examples/gallery/sepia_brightness_20_lena.png)

## Usage Examples

```bash
# Apply single modifier
mimg image.png sepia --output sepia_image.png

# Chain multiple modifiers
mimg image.png brightness 20 contrast 1.2 sharpen --output enhanced.png

# Generate this gallery
zig build gallery
```

## Available Modifiers

| Modifier | Description | Parameters |
|----------|-------------|------------|
| invert | Invert colors | None |
| grayscale | Convert to grayscale | None |
| brightness | Adjust brightness | value (-128 to 127) |
| contrast | Adjust contrast | factor (float) |
| saturation | Adjust saturation | factor (float) |
| gamma | Apply gamma correction | value (float) |
| sepia | Apply sepia tone | None |
| blur | Apply box blur | kernel_size (odd integer) |
| sharpen | Sharpen image | None |
| flip | Flip horizontally/vertically | direction |
| rotate | Rotate by angle | degrees (float) |
| resize | Resize image | width height |
| crop | Crop image | x y width height |
