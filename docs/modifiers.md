# Available Modifiers (42 Total)

## Options

| Option | Description |
|--------|-------------|
| `-o <file>` | Output filename |
| `-d <dir>` | Output directory |
| `-v` | Verbose output |
| `--preset <file>` | Load modifier chain from preset file |
| `--help` | Show help message |

## Color Adjustments (21 modifiers)

| Modifier | Usage | Description |
|----------|-------|-------------|
| `brightness` | `brightness <value>` | Adjust brightness (-128 to 127) |
| `contrast` | `contrast <factor>` | Adjust contrast |
| `saturation` | `saturation <factor>` | Adjust color saturation |
| `gamma` | `gamma <value>` | Apply gamma correction |
| `vibrance` | `vibrance <factor>` | Adjust vibrance (smart saturation) |
| `exposure` | `exposure <value>` | Adjust exposure (-2.0 to 2.0) |
| `hue-shift` | `hue-shift <degrees>` | Shift hue (-180 to 180) |
| `adjust-hsl` | `adjust-hsl <hue> <sat> <light>` | Adjust hue, saturation, and lightness separately |
| `adjust-channels` | `adjust-channels <red> <green> <blue>` | Adjust RGB channel intensities |
| `colorize` | `colorize <#RRGGBB> <intensity>` | Tint image with hex color |
| `duotone` | `duotone <#dark> <#light>` | Apply duotone effect (Spotify-style) using hex colors |
| `posterize` | `posterize <levels>` | Reduce color levels (2-256) |
| `threshold` | `threshold <value>` | Convert to black/white based on luminance (0-255) |
| `solarize` | `solarize <threshold>` | Invert colors above threshold (0-255) |
| `equalize` | `equalize` | Apply histogram equalization for better contrast |
| `equalize-area` | `equalize-area <x> <y> <width> <height>` | Apply histogram equalization to specific region |
| `grayscale` | `grayscale` | Convert to grayscale |
| `sepia` | `sepia` | Apply sepia tone |
| `invert` | `invert` | Invert colors |

## Filters (15 modifiers)

| Modifier | Usage | Description |
|----------|-------|-------------|
| `blur` | `blur <size>` | Box blur (kernel size, odd numbers) |
| `gaussian-blur` | `gaussian-blur <sigma>` | Gaussian blur with configurable sigma |
| `sharpen` | `sharpen` | Sharpen image |
| `emboss` | `emboss <strength>` | Emboss effect for 3D-like appearance |
| `color-emboss` | `color-emboss <strength>` | Emboss effect with color preservation |
| `edge-detect` | `edge-detect` | Detect edges using Sobel operator |
| `edge-enhancement` | `edge-enhancement <strength>` | Enhance edges (0.0-2.0) |
| `median-filter` | `median-filter <size>` | Median filter for noise reduction (odd kernel size) |
| `denoise` | `denoise <strength>` | Remove noise using bilateral filter (1-10) |
| `pixelate` | `pixelate <size>` | Apply pixelation/mosaic effect |
| `oil-painting` | `oil-painting <radius>` | Apply oil painting artistic effect |
| `vignette` | `vignette <intensity>` | Apply vignette effect to darken corners (0.0-1.0) |
| `glow` | `glow <intensity> <radius>` | Add soft glow around bright areas |
| `tilt-shift` | `tilt-shift <blur> <focus_pos> <focus_width>` | Advanced tilt-shift effect with two-pass Gaussian blur, smooth focus transitions, and subtle saturation boost for realistic miniature appearance. Blur: 0.0-10.0, Focus position: 0.0-1.0 (top to bottom), Focus width: 0.0-1.0 |
| `noise` | `noise <amount>` | Add random noise to image (0.0-1.0) |
| `gradient-linear` | `gradient-linear <start_color> <end_color> <angle> <opacity>` | Apply linear gradient overlay |
| `gradient-radial` | `gradient-radial <cx> <cy> <start_color> <end_color> <radius> <opacity>` | Apply radial gradient overlay |
| `censor` | `censor <x> <y> <width> <height> <method> <strength>` | Apply censoring effect (blur/pixelate/black) |

## Transforms (6 modifiers)

| Modifier | Usage | Description |
|----------|-------|-------------|
| `resize` | `resize <width> <height>` | Resize image using nearest-neighbor sampling |
| `crop` | `crop <x> <y> <width> <height>` | Crop image using top-left coordinate and size |
| `rotate` | `rotate <degrees>` | Rotate image clockwise by any angle (auto-resizes canvas) |
| `flip` | `flip <horizontal\|vertical>` | Flip image horizontally or vertically |
| `round-corners` | `round-corners <radius>` | Round corners with specified radius |

## Usage Examples

```bash
# Single modifier
zig build run -- input.png brightness 20 -o output.png

# Chain multiple modifiers
zig build run -- input.png brightness 10 contrast 1.2 sharpen -o output.png

# Advanced color grading
zig build run -- photo.jpg vibrance 0.3 exposure 0.2 contrast 1.1 hue-shift 15 -o graded.png

# Artistic effects
zig build run -- photo.jpg oil-painting 3 posterize 8 vignette 0.4 -o artistic.png

# Professional portrait enhancement
zig build run -- portrait.jpg median-filter 3 vibrance 0.2 contrast 1.1 sharpen -o enhanced.png

# Batch processing
zig build run -- *.jpg grayscale posterize 6 -d output/

# Creative duotone effect
zig build run -- photo.jpg duotone #333333 #C8C8C8 contrast 1.2 -o duotone.png

# Edge detection for comic effect
zig build run -- photo.jpg edge-detect posterize 6 contrast 1.5 -o comic.png
```

## Supported Formats
- **Input**: PNG, TGA, QOI, PAM, PBM, PGM, PPM, PCX
- **Output**: PNG, TGA, QOI, PAM, PBM, PGM, PPM, PCX</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\modifiers.md