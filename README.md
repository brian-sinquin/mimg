# mimg

[![mimg](https://github.com/brian-sinquin/mimg/actions/workflows/dev.yml/badge.svg)](https://github.com/brian-sinquin/mimg/actions/workflows/dev.yml) [![GitHub release (latest by date)](https://img.shields.io/github/v/release/brian-sinquin/mimg?label=version)](https://github.com/brian-sinquin/mimg/releases)

Command-line image processing tool written in Zig.

## Features

- Color adjustments, filters, effects, and geometric transforms
- Chain multiple modifiers in a single command
- PNG support via [`zigimg`](https://github.com/zigimg/zigimg)

## Quick Start

```cmd
# Apply a single modifier
zig build run -- input.png grayscale -o output.png

# Chain multiple modifiers
zig build run -- input.png brightness 20 saturation 1.3 sharpen -o output.png

# Complex pipeline
zig build run -- input.png exposure 0.5 vibrance 0.3 vignette 0.4 -o output.png
```

## Available Modifiers

### Color Adjustments
| Modifier | Parameters | Description |
|----------|------------|-------------|
| `brightness` | `<value>` | Linear brightness adjustment (-255 to 255) |
| `contrast` | `<factor>` | Contrast adjustment (0.0 to 3.0) |
| `saturation` | `<factor>` | Saturation adjustment (0.0 to 3.0) |
| `hue-shift` | `<degrees>` | Rotate hue in color space (0 to 360) |
| `gamma` | `<value>` | Gamma correction (0.1 to 3.0) |
| `exposure` | `<stops>` | Exposure adjustment in stops (-2.0 to 2.0) |
| `vibrance` | `<amount>` | Enhance muted colors (0.0 to 1.0) |
| `equalize` | - | Histogram equalization for contrast |

### Color Effects
| Modifier | Parameters | Description |
|----------|------------|-------------|
| `grayscale` | - | Luminance-based grayscale conversion |
| `sepia` | - | Apply sepia tone effect |
| `invert` | - | Invert all color channels |
| `threshold` | `<value>` | Binary threshold (0 to 255) |
| `solarize` | `<threshold>` | Solarization effect (0 to 255) |
| `posterize` | `<levels>` | Reduce color levels (2 to 16) |
| `colorize` | `<r> <g> <b> <strength>` | Tint with RGB color (0-255, strength 0.0-1.0) |
| `duotone` | `<r1> <g1> <b1> <r2> <g2> <b2>` | Map shadows to highlights gradient |

### Filters & Effects
| Modifier | Parameters | Description |
|----------|------------|-------------|
| `blur` | `<kernel>` | Box blur (3, 5, 7, 9, etc.) |
| `gaussian-blur` | `<sigma>` | Gaussian blur (0.5 to 5.0) |
| `sharpen` | - | 3×3 sharpen convolution |
| `edge-detect` | - | Sobel edge detection |
| `emboss` | - | 3×3 emboss effect |
| `median-filter` | `<size>` | Median filter for noise (3, 5, 7) |
| `noise` | `<amount>` | Add random noise (0.0 to 1.0) |
| `vignette` | `<strength>` | Darken corners (0.0 to 1.0) |
| `pixelate` | `<size>` | Pixelation effect (2 to 50) |
| `oil-painting` | `<radius>` | Oil painting effect (1 to 5) |

### Geometric Transforms
| Modifier | Parameters | Description |
|----------|------------|-------------|
| `resize` | `<width> <height>` | Nearest-neighbor resizing |
| `crop` | `<x> <y> <width> <height>` | Crop from top-left coordinate |
| `rotate` | `<degrees>` | Rotate clockwise (canvas auto-fits) |
| `flip-horizontal` | - | Flip image horizontally |
| `flip-vertical` | - | Flip image vertically |

### Global Options
| Option | Shorthand | Description |
|--------|-----------|-------------|
| `--output <file>` | `-o` | Output filename |
| `--output-dir <dir>` | `-d` | Output directory (auto-created) |
| `--list-modifiers` | `-L` | List all modifiers and exit |
| `--verbose` | `-v` | Enable verbose logging |
| `--help` | `-h` | Show help message |

## Usage Examples

### Basic Operations
```cmd
# Grayscale conversion
zig build run -- input.png grayscale -o output.png

# Resize to specific dimensions
zig build run -- input.png resize 800 600 -o output.png

# Crop region
zig build run -- input.png crop 50 50 300 300 -o output.png

# Rotate image
zig build run -- input.png rotate 45 -o output.png
```

### Color Adjustments
```cmd
# Brighten and increase saturation
zig build run -- input.png brightness 30 saturation 1.3 -o output.png

# Adjust contrast and gamma
zig build run -- input.png contrast 1.2 gamma 1.1 -o output.png

# Shift hue
zig build run -- input.png hue-shift 30 -o output.png

# Exposure and vibrance
zig build run -- input.png exposure 0.5 vibrance 0.4 -o output.png
```

### Creative Effects
```cmd
# Vintage look
zig build run -- input.png sepia vignette 0.4 -o output.png

# Artistic effect
zig build run -- input.png oil-painting 3 vibrance 0.3 -o output.png

# High contrast posterized
zig build run -- input.png posterize 8 contrast 1.3 -o output.png

# Duotone gradient
zig build run -- input.png duotone 75 30 120 255 220 100 -o output.png
```

### Filter Chains
```cmd
# Enhance details
zig build run -- input.png sharpen contrast 1.1 -o output.png

# Denoise and sharpen
zig build run -- input.png median-filter 3 sharpen -o output.png

# Blur and edge detect
zig build run -- input.png gaussian-blur 1.5 edge-detect -o output.png
```

### Complex Pipelines
```cmd
# Complete color correction
zig build run -- input.png exposure 0.3 contrast 1.1 saturation 1.2 vibrance 0.2 -o output.png

# Stylized output
zig build run -- input.png brightness 10 contrast 1.2 posterize 12 emboss -o output.png

# Geometric + color
zig build run -- input.png rotate 15 crop 100 100 400 400 brightness 20 saturation 1.3 -o output.png
```

## Building

### Requirements
- Zig 0.15.1 or later

### Compile
```cmd
zig build
```

The executable will be created in `zig-out/bin/`.

### Run Tests
```cmd
zig build test
```

### Generate Gallery
```cmd
zig build gallery
```

Generates visual examples in `examples/gallery/gallery.md`.

## Installation

### From Release
Download the latest binary from [releases](https://github.com/brian-sinquin/mimg/releases).

### From Source
```cmd
git clone https://github.com/brian-sinquin/mimg.git
cd mimg
zig build
```

## Examples Gallery

Visual examples with sample outputs: [examples/gallery/gallery.md](examples/gallery/gallery.md)

## License

MIT License - see [LICENSE](LICENSE) file for details.
