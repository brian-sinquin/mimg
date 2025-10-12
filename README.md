# mimg

[![mimg](https://github.com/brian-sinquin/mimg/actions/workflows/dev.yml/badge.svg)](https://github.com/brian-sinquin/mimg/actions/workflows/dev.yml) [![GitHub release (latest by date)](https://img.shields.io/github/v/release/brian-sinquin/mimg?label=ver)](https://github.com/brian-sinquin/mimg/releases/latest)

### Benchmark Results (Intel i7-9750H, 256x256 images)

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Grayscale | 995 MPixels/sec | Simple luminance calculation |
| Invert Colors | 1120 MPixels/sec | Fast bitwise operations |
| Brightness | 811 MPixels/sec | SIMD-optimized |
| Contrast | 475 MPixels/sec | Non-linear transformation |
| Sepia | 860 MPixels/sec | Matrix multiplication |
| Box Blur (3×3) | 98 MPixels/sec | Convolution kernel |
| Gaussian Blur (σ=1.0) | 76 MPixels/sec | Separable kernel |
| Sharpen | 61 MPixels/sec | Edge enhancement |
| Median Filter (3×3) | 42 MPixels/sec | Sorting operations |
| Flip Horizontal | 3597 MPixels/sec | Memory copy |
| Rotate 90° | 1368 MPixels/sec | Pixel rearrangement |github.com/brian-sinquin/mimg/releases)

Command-line image processing tool written in Zig.

## Features

- Color adjustments, filters, effects, and geometric transforms
- Chain multiple modifiers in a single command
- Save and reuse modifier chains with preset files
- Multiple image formats via [`zigimg`](https://github.com/zigimg/zigimg) (PNG, TGA, QOI, PAM, PBM, PGM, PPM, PCX)

## Quick Start

```cmd
# Apply a single modifier
zig build run -- input.png grayscale -o output.png

# Chain multiple modifiers
zig build run -- input.png brightness 20 saturation 1.3 sharpen -o output.png

# Use a preset file for complex processing
zig build run -- input.png --preset vintage.preset -o output.png

# Complex pipeline
zig build run -- input.png exposure 0.5 vibrance 0.3 vignette 0.4 -o output.png
```

## Available Modifiers

### Color Adjustments
| Modifier | Parameters | Description |
|----------|------------|-------------|
| `brightness` | `<value>` | Linear brightness adjustment (-255 to 255, default 0) |
| `contrast` | `<factor>` | Contrast adjustment (0.0 to 3.0, default 1.0) |
| `saturation` | `<factor>` | Saturation adjustment (0.0 to 3.0, default 1.0) |
| `hue-shift` | `<degrees>` | Rotate hue in color space (0 to 360, default 0) |
| `gamma` | `<value>` | Gamma correction (0.1 to 3.0, default 1.0) |
| `exposure` | `<stops>` | Exposure adjustment in stops (-2.0 to 2.0, default 0) |
| `vibrance` | `<amount>` | Enhance muted colors (0.0 to 1.0, default 0) |
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
| `blur` | `<kernel>` | Box blur (3, 5, 7, 9, etc., must be odd) |
| `gaussian-blur` | `<sigma>` | Gaussian blur (0.5 to 5.0, default 1.0) |
| `sharpen` | - | 3×3 sharpen convolution |
| `edge-detect` | - | Sobel edge detection |
| `emboss` | - | 3×3 emboss effect |
| `median-filter` | `<size>` | Median filter for noise (3, 5, 7, must be odd) |
| `noise` | `<amount>` | Add random noise (0.0 to 1.0, default 0.1) |
| `vignette` | `<strength>` | Darken corners (0.0 to 1.0, default 0.5) |
| `pixelate` | `<size>` | Pixelation effect (2 to 50, default 8) |
| `oil-painting` | `<radius>` | Oil painting effect (1 to 5, default 3) |

### Geometric Transforms
| Modifier | Parameters | Description |
|----------|------------|-------------|
| `resize` | `<width> <height>` | Nearest-neighbor resizing (max 65535×65535) |
| `crop` | `<x> <y> <width> <height>` | Crop from top-left coordinate (all ≥ 0) |
| `rotate` | `<degrees>` | Rotate clockwise (90, 180, 270 only) |
| `flip-horizontal` | - | Flip image horizontally |
| `flip-vertical` | - | Flip image vertically |

### Global Options
| Option | Shorthand | Description |
|--------|-----------|-------------|
| `--output <file>` | `-o` | Output filename |
| `--output-dir <dir>` | `-d` | Output directory (auto-created) |
| `--output-extension <ext>` | - | Output file extension (e.g., .png, .jpg, .tga) |
| `--preset <file>` | `-p` | Load modifier chain from preset file |
| `--list-modifiers` | `-L` | List all modifiers and exit |
| `--verbose` | `-v` | Enable verbose logging |
| `--help` | `-h` | Show help message |

## Presets

Presets allow you to save and reuse chains of modifiers. Create a text file with one modifier per line, then use the `--preset` option to apply the entire chain.

### Creating Presets

Create a text file (e.g., `vintage.preset`) with modifier commands:

```
# Vintage photo effect
sepia
vignette 0.4
contrast 1.1
brightness 10
```

Or a complex processing chain:

```
# Professional color correction
exposure 0.3
contrast 1.2
saturation 1.1
vibrance 0.2
sharpen
```

### Using Presets

Apply a preset to any image:

```cmd
# Apply vintage preset
zig build run -- input.png --preset vintage.preset -o output.png

# Combine preset with additional modifiers
zig build run -- input.png --preset color-correction.preset gamma 1.1 -o output.png

# Use preset in batch processing
zig build run -- *.png --preset batch-process.preset -d output/
```

### Preset File Format

- One modifier per line
- Parameters separated by spaces
- Empty lines and lines starting with `#` are ignored
- Comments can be added with `#`

Example preset file:

```
# HDR-like effect
exposure 0.5
contrast 1.3
vibrance 0.3
gamma 1.2

# Add some sharpening
sharpen
```

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

# Using presets for batch processing
zig build run -- *.png --preset professional-grade.preset -d output/
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

## Performance

mimg is optimized for speed with SIMD vector operations and efficient algorithms. Performance varies by image size and operation complexity.

### Benchmark Results (Intel i7-9750H, 256x256 images)

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Grayscale | 1078 MPixels/sec | Simple luminance calculation |
| Invert Colors | 1025 MPixels/sec | Fast bitwise operations |
| Brightness | 831 MPixels/sec | SIMD-optimized |
| Contrast | 492 MPixels/sec | Non-linear transformation |
| Sepia | 880 MPixels/sec | Matrix multiplication |
| Box Blur (3×3) | 93 MPixels/sec | Convolution kernel |
| Gaussian Blur (σ=1.0) | 72 MPixels/sec | Separable kernel |
| Sharpen | 56 MPixels/sec | Edge enhancement |
| Median Filter (3×3) | 39 MPixels/sec | Sorting operations |
| Flip Horizontal | 3600 MPixels/sec | Memory copy |
| Rotate 90° | 1334 MPixels/sec | Pixel rearrangement |

### Scaling Performance

- **Small images** (< 1000×1000): Fast processing, suitable for batch operations
- **Large images** (> 4000×4000): Memory-efficient streaming, may be slower for complex filters
- **Batch processing**: Multi-threaded for multiple files
- **Memory usage**: ~4 bytes per pixel (RGBA32) + temporary buffers for filters

### Memory Usage & Optimizations

mimg is optimized for memory efficiency with intelligent buffer reuse and tiled processing for large images:

- **Buffer Reuse**: Temporary buffers are automatically reused across operations to minimize allocations
- **Tiled Processing**: Large images are processed in tiles for memory-intensive filters (median, oil painting)
- **Memory Limits**: Maximum 50M pixels (~200MB for RGBA32) to prevent excessive memory usage
- **Automatic Cleanup**: All temporary buffers are properly managed and freed

### Memory Usage Examples

- **Simple operations** (grayscale, invert): ~4 bytes per pixel (image buffer only)
- **Filters with temp buffers** (blur, sharpen): ~8 bytes per pixel (image + temp buffer)
- **Complex operations** (median, oil painting): Variable, uses tiled processing for large images
- **Transforms** (resize, rotate): Allocates new buffer for result, old buffer freed automatically

Use `--verbose` flag to see detailed memory usage information during processing.

## Error Handling

mimg provides descriptive error messages to help troubleshoot issues:

### Common Errors

- **"Image too large"**: Image exceeds maximum dimensions (65535×65535) or pixel count (50M pixels)
  - Solution: Resize large images before processing or use smaller regions

- **"Invalid parameter range"**: Parameter value outside allowed range
  - Check parameter ranges in the modifier tables above
  - Example: `contrast 5.0` → use `contrast 3.0` (maximum)

- **"Kernel size must be odd"**: Blur/median filters require odd kernel sizes
  - Use 3, 5, 7, 9, etc. instead of even numbers

- **"Crop rectangle out of bounds"**: Crop coordinates exceed image dimensions
  - Ensure x+width ≤ image_width and y+height ≤ image_height

- **"Unsupported rotation angle"**: Only 90°, 180°, 270° rotations supported
  - Use multiple 90° rotations for other angles

### Memory Considerations

- Large images (> 4000×4000) may require significant RAM
- Filters create temporary buffers (additional ~4 bytes per pixel)
- For very large images, consider processing in tiles or resizing first

### File Format Support

mimg supports multiple image formats for both input and output via the [`zigimg`](https://github.com/zigimg/zigimg) library:

#### Supported Output Formats
- **PNG**: Portable Network Graphics (lossless, default)
- **TGA**: Truevision TGA (lossless)
- **QOI**: Quite OK Image (lossless, fast)
- **PAM**: Portable Arbitrary Map (lossless)
- **PBM/PGM/PPM**: NetPBM formats (grayscale/binary variants)
- **PCX**: ZSoft PCX (legacy format)

#### Supported Input Formats
mimg can read any format supported by zigimg, including:
- PNG, JPEG, BMP, TGA, QOI, PAM, PBM, PGM, PPM, PCX, and more
- URL support for remote images (http/https)

Use the `--output-extension` option to specify the output format:

```cmd
# Save as different formats
zig build run -- input.jpg --output-extension .png -o output.png
zig build run -- input.png --output-extension .tga -o output.tga
zig build run -- input.jpg --output-extension .qoi -o output.qoi
```

## Troubleshooting

### Performance Issues

- **Slow processing**: Use smaller kernel sizes, apply transforms before filters
- **High memory usage**: Resize large images or process in smaller batches
- **Batch processing slow**: Use presets to avoid argument parsing overhead

### Quality Issues

- **Artifacts in filters**: Reduce filter strength or apply multiple passes
- **Color banding**: Use higher bit depth or dithering (not yet implemented)
- **Edge artifacts**: Apply filters with smaller kernels

### Build Issues

- **Zig version mismatch**: Requires Zig 0.15.1 or later
- **Missing dependencies**: Run `zig build` to fetch dependencies automatically
- **Test failures**: Ensure all source files are present and unmodified

### Getting Help

- Check parameter ranges in modifier tables
- Use `--verbose` flag for detailed processing information
- Test with small images first to isolate issues
- Report bugs with sample input and exact command used

## Library Usage

mimg can be used as a Zig library for programmatic image processing. Add mimg as a dependency in your `build.zig.zon`:

```zig
.dependencies = .{
    .mimg = .{
        .url = "https://github.com/brian-sinquin/mimg/archive/main.tar.gz",
        .hash = "...", // Get from zig fetch
    },
},
```

### Basic Usage

```zig
const std = @import("std");
const img = @import("zigimg");
const mimg = @import("mimg");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create context
    var ctx = mimg.types.Context.init(allocator);
    defer ctx.deinit();

    // Load image
    const image = try img.Image.fromFilePath(allocator, "input.png");
    defer image.deinit(allocator);
    ctx.setImage(image);

    // Apply processing
    try mimg.basic.adjustBrightness(&ctx, .{20});      // +20 brightness
    try mimg.basic.adjustContrast(&ctx, .{1.2});       // 20% more contrast
    try mimg.basic.gaussianBlurImage(&ctx, .{1.0});    // Gaussian blur σ=1.0

    // Save result
    try ctx.image.writeToFilePath("output.png", .{});
}
```

### Context Management

The `Context` struct manages image state and temporary buffers:

```zig
var ctx = mimg.types.Context.init(allocator);
defer ctx.deinit();

// Load image
const image = try img.Image.fromFilePath(allocator, "input.png");
ctx.setImage(image);

// Context automatically manages memory and temporary buffers
```

### Available Functions

All processing functions follow the pattern `functionName(ctx: *Context, args: anytype) !void`:

#### Color Adjustments
- `invertColors(ctx, .{})` - Invert all colors
- `grayscaleImage(ctx, .{})` - Convert to grayscale
- `adjustBrightness(ctx, .{value})` - Adjust brightness (-255 to 255)
- `adjustContrast(ctx, .{factor})` - Adjust contrast (0.0 to 3.0)
- `adjustSaturation(ctx, .{factor})` - Adjust saturation (0.0 to 3.0)
- `adjustGamma(ctx, .{value})` - Gamma correction (0.1 to 3.0)
- `hueShiftImage(ctx, .{degrees})` - Hue rotation (0 to 360)
- `applySepia(ctx, .{})` - Apply sepia tone

#### Filters
- `gaussianBlurImage(ctx, .{sigma})` - Gaussian blur (0.5 to 5.0)
- `sharpenImage(ctx, .{})` - Sharpen image
- `medianFilterImage(ctx, .{size})` - Median filter (3, 5, 7)
- `vignetteImage(ctx, .{strength})` - Add vignette (0.0 to 1.0)

#### Transforms
- `resizeImage(ctx, .{width, height})` - Resize image
- `cropImage(ctx, .{x, y, width, height})` - Crop rectangle
- `flipImage(ctx, .{"horizontal"})` - Flip horizontally
- `rotateImage(ctx, .{degrees})` - Rotate (90, 180, 270)

### Error Handling

mimg uses Zig's error union types for robust error handling:

```zig
const result = mimg.basic.adjustBrightness(&ctx, .{300});
if (result) {
    // Success
} else |err| switch (err) {
    error.InvalidParameters => std.debug.print("Invalid brightness value\n", .{}),
    error.OutOfMemory => std.debug.print("Not enough memory\n", .{}),
    else => std.debug.print("Processing error: {}\n", .{err}),
}
```

Common error types:
- `ProcessingError` - Invalid parameters, unsupported operations
- `ImageError` - File I/O, format issues, memory problems
- `FileSystemError` - Path, permission, disk issues

### Memory Management

mimg automatically manages temporary buffers for filters. For large images or batch processing:

```zig
// Context reuses temporary buffers automatically
// No manual buffer management needed for most use cases

// For very large images, consider processing in sections
// or resizing before applying memory-intensive filters
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
