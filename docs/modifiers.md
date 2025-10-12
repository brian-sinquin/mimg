# Available Modifiers

mimg provides 30+ image processing modifiers organized into categories. Each modifier can be chained together for complex processing pipelines.

## Global Options

| Option | Shorthand | Description |
|--------|-----------|-------------|
| `--output <file>` | `-o` | Output filename |
| `--output-dir <dir>` | `-d` | Output directory (auto-created) |
| `--output-extension <ext>` | - | Output file extension (e.g., .png, .jpg, .tga) |
| `--preset <file>` | `-p` | Load modifier chain from preset file |
| `--list-modifiers` | `-L` | List all modifiers and exit |
| `--verbose` | `-v` | Enable verbose logging |
| `--help` | `-h` | Show help message |

## Color Adjustments

Fine-tune image colors and tones.

| Modifier | Parameters | Range | Description |
|----------|------------|-------|-------------|
| `brightness` | `<value>` | -255 to 255 | Linear brightness adjustment |
| `contrast` | `<factor>` | 0.0 to 3.0 | Contrast adjustment |
| `saturation` | `<factor>` | 0.0 to 3.0 | Saturation adjustment |
| `hue-shift` | `<degrees>` | 0 to 360 | Rotate hue in color space |
| `gamma` | `<value>` | 0.1 to 3.0 | Gamma correction |
| `exposure` | `<stops>` | -2.0 to 2.0 | Exposure adjustment in stops |
| `vibrance` | `<amount>` | 0.0 to 1.0 | Enhance muted colors |
| `equalize` | - | - | Histogram equalization for contrast |

**Examples:**
```bash
# Basic color correction
zig build run -- input.png brightness 20 contrast 1.2 saturation 1.1 -o output.png

# Creative color effects
zig build run -- input.png hue-shift 45 vibrance 0.3 -o output.png
```

## Color Effects

Transform colors with artistic effects.

| Modifier | Parameters | Range | Description |
|----------|------------|-------|-------------|
| `grayscale` | - | - | Luminance-based grayscale conversion |
| `sepia` | - | - | Apply sepia tone effect |
| `invert` | - | - | Invert all color channels |
| `threshold` | `<value>` | 0 to 255 | Binary threshold |
| `solarize` | `<threshold>` | 0 to 255 | Solarization effect |
| `posterize` | `<levels>` | 2 to 16 | Reduce color levels |
| `colorize` | `<r> <g> <b> <strength>` | 0-255, 0.0-1.0 | Tint with RGB color |
| `duotone` | `<r1> <g1> <b1> <r2> <g2> <b2>` | 0-255 | Map shadows to highlights gradient |

**Examples:**
```bash
# Classic effects
zig build run -- input.png sepia -o sepia.png
zig build run -- input.png grayscale -o gray.png

# Advanced color manipulation
zig build run -- input.png posterize 8 -o poster.png
zig build run -- input.png duotone 75 30 120 255 220 100 -o duotone.png
```

## Filters & Effects

Apply convolution filters and artistic effects.

| Modifier | Parameters | Range | Description |
|----------|------------|-------|-------------|
| `blur` | `<kernel>` | 3,5,7,9... (odd) | Box blur |
| `gaussian-blur` | `<sigma>` | 0.5 to 5.0 | Gaussian blur |
| `sharpen` | - | - | 3×3 sharpen convolution |
| `edge-detect` | - | - | Sobel edge detection |
| `emboss` | - | - | 3×3 emboss effect |
| `median-filter` | `<size>` | 3,5,7 (odd) | Median filter for noise |
| `noise` | `<amount>` | 0.0 to 1.0 | Add random noise |
| `vignette` | `<strength>` | 0.0 to 1.0 | Darken corners |
| `pixelate` | `<size>` | 2 to 50 | Pixelation effect |
| `oil-painting` | `<radius>` | 1 to 5 | Oil painting effect |

**Examples:**
```bash
# Basic filtering
zig build run -- input.png gaussian-blur 1.5 -o blur.png
zig build run -- input.png sharpen -o sharp.png

# Artistic effects
zig build run -- input.png oil-painting 3 -o oil.png
zig build run -- input.png vignette 0.5 -o vignette.png
```

## Geometric Transforms

Change image dimensions and orientation.

| Modifier | Parameters | Range | Description |
|----------|------------|-------|-------------|
| `resize` | `<width> <height>` | 1 to 65535 | Nearest-neighbor resizing |
| `crop` | `<x> <y> <width> <height>` | ≥ 0 | Crop from top-left coordinate |
| `rotate` | `<degrees>` | 90, 180, 270 | Rotate clockwise |
| `flip-horizontal` | - | - | Flip image horizontally |
| `flip-vertical` | - | - | Flip image vertically |

**Examples:**
```bash
# Basic transforms
zig build run -- input.png resize 800 600 -o resized.png
zig build run -- input.png rotate 90 -o rotated.png

# Combined operations
zig build run -- input.png crop 100 100 400 400 flip-horizontal -o cropped.png
```

## Modifier Chaining

Modifiers are applied in the order specified on the command line. Each modifier operates on the result of the previous one.

**Syntax:**
```bash
zig build run -- input.png [modifier1] [params...] [modifier2] [params...] -o output.png
```

**Examples:**
```bash
# Color correction pipeline
zig build run -- input.png brightness 10 contrast 1.2 saturation 1.1 -o corrected.png

# Creative effect chain
zig build run -- input.png sepia vignette 0.4 contrast 1.1 -o vintage.png

# Complex processing
zig build run -- input.png resize 1024 768 gaussian-blur 0.8 sharpen exposure 0.3 -o processed.png
```

## Parameter Validation

mimg validates all parameters and provides helpful error messages:

- **Range checking**: Parameters must be within specified ranges
- **Type validation**: Numeric parameters only where expected
- **Image bounds**: Crop/resize operations check image dimensions
- **Kernel sizes**: Must be odd numbers for convolution filters

**Example error messages:**
```
Error: brightness value must be between -255 and 255
Error: kernel size must be odd (3, 5, 7, etc.)
Error: crop rectangle (100, 100, 500, 500) exceeds image bounds (400, 300)
```

## Performance Notes

- **SIMD Optimization**: Color adjustments use SIMD for maximum speed
- **Memory Efficiency**: Filters reuse buffers to minimize allocations
- **Tiled Processing**: Large images processed in tiles for memory-intensive operations
- **Fast Operations**: Geometric transforms are highly optimized

## Next Steps

- [Presets](presets.md) - Save and reuse modifier chains
- [Examples](examples.md) - Practical usage examples
- [Performance](performance.md) - Benchmarks and optimization tips</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\modifiers.md