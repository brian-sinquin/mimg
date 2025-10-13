# 🎨 MIMG Gallery

Visual showcase of all available image processing modifiers.

*All examples generated from the standard `lena.png` test image. For complete documentation, see the [main README](../../README.md).*

---

## Original Image

![Original](lena.png)

---

## Individual Modifiers

### Invert

Invert all colors

![Invert](output/invert_lena.png)

### Grayscale

Convert to grayscale

![Grayscale](output/grayscale_lena.png)

### Brightness +50

Increase brightness

![Brightness +50](output/brightness_50_lena.png)

### Brightness -30

Decrease brightness

![Brightness -30](output/brightness_-30_lena.png)

### Contrast 1.5

Increase contrast

![Contrast 1.5](output/contrast_1_5_lena.png)

### Contrast 0.7

Decrease contrast

![Contrast 0.7](output/contrast_0_7_lena.png)

### Saturation 1.5

Increase saturation

![Saturation 1.5](output/saturation_1_5_lena.png)

### Saturation 0.5

Decrease saturation

![Saturation 0.5](output/saturation_0_5_lena.png)

### Gamma 2.2

Apply gamma correction (brighter)

![Gamma 2.2](output/gamma_2_2_lena.png)

### Gamma 0.45

Apply gamma correction (darker)

![Gamma 0.45](output/gamma_0_45_lena.png)

### Sepia

Apply sepia tone effect

![Sepia](output/sepia_lena.png)

### Blur

Apply box blur with kernel size 3

![Blur](output/blur_3_lena.png)

### Sharpen

Sharpen the image

![Sharpen](output/sharpen_lena.png)

### Gaussian Blur

Apply Gaussian blur with sigma 2.0

![Gaussian Blur](output/gaussian-blur_2_0_lena.png)

### Emboss

Apply emboss effect

![Emboss](output/emboss_lena.png)

### Vignette

Apply vignette effect

![Vignette](output/vignette_0_5_lena.png)

### Posterize

Reduce to 8 color levels

![Posterize](output/posterize_8_lena.png)

### Hue Shift

Shift hue by 15 degrees

![Hue Shift](output/hue-shift_15_lena.png)

### Median Filter

Apply 3x3 median filter

![Median Filter](output/median-filter_3_lena.png)

### Threshold

Convert to pure black and white

![Threshold](output/threshold_128_lena.png)

### Solarize

Solarize effect with threshold

![Solarize](output/solarize_128_lena.png)

### Edge Detect

Detect edges using Sobel operator

![Edge Detect](output/edge-detect_lena.png)

### Pixelate

Apply pixelation effect

![Pixelate](output/pixelate_10_lena.png)

### Noise

Add random noise to image

![Noise](output/noise_0_1_lena.png)

### Exposure +1

Increase exposure by 1 EV

![Exposure +1](output/exposure_1_0_lena.png)

### Exposure -1

Decrease exposure by 1 EV

![Exposure -1](output/exposure_-1_0_lena.png)

### Vibrance

Boost vibrance (smart saturation)

![Vibrance](output/vibrance_0_5_lena.png)

### Equalize

Histogram equalization for contrast

![Equalize](output/equalize_lena.png)

### Colorize Blue

Tint with blue color

![Colorize Blue](output/colorize_50_100_200_0_6_lena.png)

### Duotone

Purple to yellow duotone effect

![Duotone](output/duotone_75_30_120_255_220_100_lena.png)

### Oil Painting

Artistic oil painting effect

![Oil Painting](output/oil-painting_3_lena.png)

### Flip Horizontal

Mirror horizontally

![Flip Horizontal](output/flip_horizontal_lena.png)

### Flip Vertical

Mirror vertically

![Flip Vertical](output/flip_vertical_lena.png)

### Rotate 45°

Rotate 45 degrees clockwise

![Rotate 45°](output/rotate_45_lena.png)

### Resize

Resize to 200x200

![Resize](output/resize_200_200_lena.png)

### Crop

Crop to center 200x200

![Crop](output/crop_25_25_200_200_lena.png)


---

## Creative Combinations

### Data Augmentation

Flip + brightness variation for ML training data

![Data Augmentation](output/flip_horizontal_brightness_20_lena.png)

### Geometric Augmentation

Rotation + crop for dataset variety

![Geometric Augmentation](output/rotate_15_crop_50_50_200_200_lena.png)

### Vintage Portrait

Sepia tone with vignette for classic look

![Vintage Portrait](output/sepia_vignette_0_4_lena.png)

### Color Grading

Brightness + saturation for natural enhancement

![Color Grading](output/brightness_15_saturation_1_2_lena.png)

### Detail Enhancement

Contrast boost + sharpening for crisp images

![Detail Enhancement](output/contrast_1_2_sharpen_lena.png)

### Noise Reduction

Gaussian blur + median filter pipeline

![Noise Reduction](output/gaussian-blur_1_0_median-filter_3_lena.png)

### Graphic Art

Posterize + emboss for artistic rendering

![Graphic Art](output/posterize_12_emboss_lena.png)

### Monitor Calibration

Gamma correction + contrast adjustment

![Monitor Calibration](output/gamma_2_2_contrast_1_1_lena.png)

### Stylized Edges

Edge detection + posterize for graphic novel style

![Stylized Edges](output/edge-detect_posterize_6_lena.png)

### Retro Game

Pixelate + posterize for retro gaming aesthetic

![Retro Game](output/pixelate_8_posterize_16_lena.png)

### Film Grain

Solarize + noise for vintage film look

![Film Grain](output/solarize_180_noise_0_05_lena.png)

### HDR Look

Equalize + vibrance for HDR-style enhancement

![HDR Look](output/equalize_vibrance_0_3_lena.png)

### Dreamy Portrait

Oil painting + vibrance for soft romantic look

![Dreamy Portrait](output/oil-painting_2_vibrance_0_4_lena.png)

### Cinematic Grade

Duotone + exposure for film-like color

![Cinematic Grade](output/duotone_20_40_80_255_200_150_exposure_0_2_lena.png)


---

<p align="center">
  <i>Generated automatically by <code>zig build gallery</code></i><br>
  <i>See <a href="../../README.md">README.md</a> for full documentation and usage examples</i>
</p>
