# File Formats

mimg supports multiple image formats through the [`zigimg`](https://github.com/zigimg/zigimg) library. This guide covers supported formats, format selection, and format-specific considerations.

## Supported Output Formats

mimg can save images in the following formats:

| Format | Extension | Compression | Quality | Use Case |
|--------|-----------|-------------|---------|----------|
| **PNG** | `.png` | Lossless | Perfect | Web, graphics, default |
| **TGA** | `.tga` | Lossless | Perfect | Games, legacy support |
| **QOI** | `.qoi` | Lossless | Perfect | Fast, modern format |
| **PAM** | `.pam` | Lossless | Perfect | NetPBM variant |
| **PBM** | `.pbm` | Lossless | Binary | Black & white |
| **PGM** | `.pgm` | Lossless | Grayscale | Grayscale images |
| **PPM** | `.ppm` | Lossless | RGB | Full color |

## Supported Input Formats

mimg can read any format supported by zigimg, including:

- **PNG, JPEG, BMP, TGA, QOI, PAM, PBM, PGM, PPM, PCX**
- **Remote images**: HTTP/HTTPS URLs supported
- **Auto-detection**: Format detected from file extension or content

## Format Selection

### Automatic Format Detection

By default, mimg preserves the input format:

```bash
# Input PNG → Output PNG
zig build run -- photo.png brightness 20 -o enhanced.png

# Input JPG → Output JPG (if supported)
zig build run -- photo.jpg brightness 20 -o enhanced.jpg
```

### Explicit Format Conversion

Use `--output-extension` to specify output format:

```bash
# Convert to PNG
zig build run -- photo.jpg --output-extension .png brightness 20 -o converted.png

# Convert to TGA
zig build run -- photo.png --output-extension .tga -o output.tga

# Convert to QOI
zig build run -- photo.jpg --output-extension .qoi -o fast.qoi
```

### Batch Format Conversion

```bash
# Convert all JPG to PNG
zig build run -- *.jpg --output-extension .png -d png-versions/

# Convert to modern formats
zig build run -- *.png --output-extension .qoi -d qoi-versions/
```

## Format Details

### PNG (Portable Network Graphics)

**Best for**: Web images, graphics with transparency, lossless quality

```bash
# High-quality PNG output
zig build run -- input.jpg --output-extension .png -o output.png
```

**Characteristics**:
- Lossless compression
- Supports transparency (alpha channel)
- Broad browser support
- Larger file sizes than lossy formats

### TGA (Truevision TGA)

**Best for**: Game development, legacy systems, uncompressed needs

```bash
# TGA for game assets
zig build run -- sprite.png --output-extension .tga -o sprite.tga
```

**Characteristics**:
- Lossless
- Supports alpha channel
- Uncompressed option available
- Common in game engines

### QOI (Quite OK Image)

**Best for**: Fast encoding/decoding, modern applications

```bash
# Fast QOI conversion
zig build run -- photo.png --output-extension .qoi -o photo.qoi
```

**Characteristics**:
- Lossless
- Very fast encoding/decoding
- Small file sizes
- Modern format with growing support

### NetPBM Formats (PAM/PBM/PGM/PPM)

**Best for**: Data exchange, debugging, specialized applications

```bash
# PAM (Portable Arbitrary Map) - full color
zig build run -- input.png --output-extension .pam -o output.pam

# PGM (Portable Gray Map) - grayscale
zig build run -- input.png grayscale --output-extension .pgm -o gray.pgm

# PBM (Portable Bitmap) - black & white
zig build run -- input.png threshold 128 --output-extension .pbm -o bw.pbm
```

**Characteristics**:
- Lossless text-based formats
- Human-readable headers
- Platform-independent
- Used in scientific computing

## Remote Image Processing

mimg can process images directly from URLs:

```bash
# Process remote image
zig build run -- https://example.com/image.png brightness 20 -o enhanced.png

# Combine with other operations
zig build run -- https://example.com/photo.jpg resize 800 600 vignette 0.3 -o processed.png
```

**Supported protocols**:
- HTTP
- HTTPS

**Limitations**:
- No authentication support
- Large images may take time to download
- Network timeouts may occur

## Format-Specific Considerations

### Transparency Handling

```bash
# PNG with transparency preserved
zig build run -- transparent.png brightness 20 -o bright-transparent.png

# TGA with alpha channel
zig build run -- rgba-image.png --output-extension .tga -o rgba.tga
```

### Color Space Considerations

- **RGB**: All formats support full color
- **Grayscale**: Optimized in PGM format
- **Binary**: PBM format for 1-bit images
- **RGBA**: PNG, TGA, PAM support alpha channels

### File Size Comparison

Typical file sizes for a 1024×768 test image:

| Format | Size | Compression | Quality |
|--------|------|-------------|---------|
| PNG | ~450KB | Lossless | Perfect |
| TGA | ~2.3MB | Uncompressed | Perfect |
| QOI | ~380KB | Lossless | Perfect |
| PAM | ~2.3MB | Uncompressed | Perfect |

## Batch Processing with Formats

### Directory-Based Conversion

```bash
# Convert entire directory
zig build run -- input-dir/*.png --output-extension .qoi -d qoi-output/

# Mixed format handling
find . -name "*.jpg" -o -name "*.png" | xargs -I {} zig build run -- {} --output-extension .qoi -o converted/{}
```

### Format-Specific Workflows

```bash
# Web optimization pipeline
zig build run -- photos/*.jpg \
  resize 1920 1080 \
  --output-extension .png \
  -d web-ready/

# Game asset pipeline
zig build run -- sprites/*.png \
  --output-extension .tga \
  -d game-assets/
```

## Format Detection Issues

### Extension Mismatch

If files have wrong extensions, mimg may fail to load them:

```bash
# File is PNG but named .jpg
# This may fail or produce unexpected results
zig build run -- misnamed.jpg brightness 20 -o output.png
```

**Solution**: Rename files or use correct extensions.

### Corrupted Files

mimg will report errors for corrupted or unsupported files:

```
Error: Failed to load image: Invalid PNG signature
Error: Unsupported image format
```

**Solutions**:
- Verify file integrity
- Check if format is supported
- Try converting with other tools first

## Performance by Format

### Encoding Speed

| Format | Speed | Notes |
|--------|-------|-------|
| QOI | Fastest | Optimized for speed |
| PNG | Medium | Good compression/speed balance |
| TGA | Fast | Minimal compression |
| PAM | Fast | No compression |

### Decoding Speed

| Format | Speed | Notes |
|--------|-------|-------|
| QOI | Fastest | Fast loading |
| PNG | Medium | Standard performance |
| TGA | Fast | Direct pixel access |
| PAM | Fast | Simple format |

## Next Steps

- [Installation](installation.md) - Get mimg with format support
- [Examples](examples.md) - Format conversion examples
- [Troubleshooting](troubleshooting.md) - Fix format-related issues</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\formats.md