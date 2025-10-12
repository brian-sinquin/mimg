# Troubleshooting

This guide helps you resolve common issues with mimg. Most problems have simple solutions related to parameters, file formats, or system configuration.

## Quick Diagnosis

### Check Your Setup

```bash
# Verify installation
zig build --help

# Check Zig version
zig version  # Should be 0.15.1 or later

# Test basic functionality
zig build run -- --help
```

### Enable Verbose Output

Add `--verbose` to see detailed processing information:

```bash
zig build run -- image.png --verbose brightness 20 -o output.png
```

This shows:
- Image dimensions and format
- Memory usage
- Processing steps
- Performance metrics

## Common Errors

### "Image too large"

**Error**: `Image dimensions exceed maximum limits`

**Cause**: Image is larger than 65535×65535 pixels or 50M total pixels

**Solutions**:
```bash
# Resize before processing
zig build run -- huge-image.jpg resize 4000 3000 -o sized.jpg

# Crop to smaller region
zig build run -- huge-image.jpg crop 0 0 4000 3000 -o cropped.jpg

# Process in tiles (automatic for complex filters)
zig build run -- huge-image.jpg median-filter 3 -o processed.jpg
```

### "Invalid parameter range"

**Error**: `Parameter value outside allowed range`

**Cause**: Modifier parameter is too high/low

**Solutions**:
```bash
# Check parameter ranges in modifier tables
zig build run -- image.png contrast 1.5 -o output.png    # Valid: 0.0-3.0
zig build run -- image.png brightness 100 -o output.png  # Valid: -255 to 255

# Use --list-modifiers to see all valid ranges
zig build run -- --list-modifiers
```

### "Kernel size must be odd"

**Error**: `Kernel size must be odd number`

**Cause**: Blur/median filters require odd kernel sizes

**Solutions**:
```bash
# Use odd numbers only
zig build run -- image.png blur 3 -o output.png        # ✓ Valid
zig build run -- image.png blur 5 -o output.png        # ✓ Valid
zig build run -- image.png blur 4 -o output.png        # ✗ Invalid

# Common valid sizes: 3, 5, 7, 9, 11
```

### "Crop rectangle out of bounds"

**Error**: `Crop coordinates exceed image dimensions`

**Cause**: Crop area extends beyond image boundaries

**Solutions**:
```bash
# Check image dimensions first
zig build run -- image.png --verbose resize 100 100 -o temp.png  # Shows dimensions

# Ensure crop area fits within image
# Image is 800×600
zig build run -- image.png crop 0 0 800 600 -o cropped.png    # ✓ Valid
zig build run -- image.png crop 100 100 700 500 -o cropped.png # ✓ Valid
zig build run -- image.png crop 100 100 800 600 -o cropped.png # ✗ Invalid (too wide)
```

### "Unsupported rotation angle"

**Error**: `Only 90°, 180°, 270° rotations supported`

**Cause**: Rotation angle not supported

**Solutions**:
```bash
# Use supported angles
zig build run -- image.png rotate 90 -o output.png   # ✓ Valid
zig build run -- image.png rotate 180 -o output.png  # ✓ Valid
zig build run -- image.png rotate 270 -o output.png  # ✓ Valid

# For other angles, rotate multiple times
zig build run -- image.png rotate 90 rotate 90 -o 180deg.png
```

## File Format Issues

### "Failed to load image"

**Error**: `Could not load image file`

**Causes & Solutions**:

```bash
# Check file exists and is readable
ls -la image.png

# Verify format is supported
file image.png  # Should show valid image format

# Try different format
zig build run -- image.jpg --output-extension .png brightness 20 -o converted.png

# Check for corruption
# Try opening in another image viewer
```

### "Unsupported image format"

**Error**: `Image format not supported`

**Solutions**:
```bash
# Convert to supported format first
convert image.bmp image.png  # Using ImageMagick
zig build run -- image.png brightness 20 -o output.png

# Supported formats: PNG, JPG, BMP, TGA, QOI, PAM, PBM, PGM, PPM, PCX
```

### Output Format Issues

```bash
# Specify output format explicitly
zig build run -- input.jpg --output-extension .png -o output.png

# Check output directory permissions
mkdir -p output/
zig build run -- input.png -d output/ brightness 20
```

## Memory Issues

### "Out of memory"

**Error**: `Failed to allocate memory`

**Solutions**:
```bash
# Process smaller images
zig build run -- large.jpg resize 2000 1500 -o smaller.jpg

# Use less memory-intensive operations
zig build run -- image.jpg blur 3 -o output.jpg  # Instead of median-filter

# Close other memory-intensive applications
# Restart with more available RAM
```

### High Memory Usage

**Monitor memory usage**:
```bash
zig build run -- large-image.jpg --verbose median-filter 5 -o output.jpg
```

**Optimize memory usage**:
```bash
# Apply transforms before filters
zig build run -- huge.jpg resize 2000 1500 gaussian-blur 1.0 -o optimized.jpg

# Use smaller filter sizes
zig build run -- image.jpg median-filter 3 -o output.jpg  # Instead of size 7
```

## Performance Issues

### Slow Processing

**Symptoms**: Operations take too long

**Diagnose**:
```bash
# Check image size
zig build run -- image.jpg --verbose brightness 20 -o output.jpg

# Time the operation
time zig build run -- large.jpg gaussian-blur 2.0 -o output.jpg
```

**Solutions**:
```bash
# Resize large images first
zig build run -- huge.jpg resize 2000 1500 gaussian-blur 1.0 -o fast.jpg

# Use faster alternatives
zig build run -- image.jpg blur 5 -o fast-blur.jpg      # Instead of gaussian-blur
zig build run -- image.jpg sharpen -o fast-sharpen.jpg  # Instead of unsharp mask

# Reduce filter strength
zig build run -- image.jpg gaussian-blur 0.5 -o subtle.jpg
```

### Batch Processing Problems

```bash
# Process in smaller batches
ls *.jpg | head -5 | xargs -I {} zig build run -- {} brightness 10 -o output/{}

# Check available disk space
df -h

# Ensure output directory exists
mkdir -p output/
```

## Build Issues

### Zig Version Problems

```bash
# Check version
zig version  # Should be 0.15.1+

# Update Zig if needed
# Download from https://ziglang.org/download/
```

### Dependency Issues

```bash
# Clean and rebuild
rm -rf .zig-cache/
zig build

# Update dependencies
zig fetch
```

### Platform-Specific Issues

**Linux**:
```bash
# Check for required libraries
ldd zig-out/bin/mimg

# Install build essentials if needed
sudo apt-get install build-essential  # Ubuntu/Debian
```

**macOS**:
```bash
# Install Xcode tools
xcode-select --install

# Check Homebrew if needed
brew install zig
```

**Windows**:
```bash
# Use official Zig build
# Ensure PATH includes Zig directory
zig version
```

## Preset Issues

### "Preset file not found"

```bash
# Check file exists
ls -la mypreset.preset

# Use absolute path
zig build run -- image.png --preset /full/path/to/preset.preset -o output.png

# Check file permissions
chmod 644 mypreset.preset
```

### "Invalid modifier in preset"

```bash
# Check preset syntax
cat mypreset.preset

# Validate modifiers
zig build run -- --list-modifiers

# Fix syntax (one modifier per line, no quotes)
echo "brightness 20
contrast 1.2" > fixed.preset
```

## Network Issues (Remote Images)

### "Failed to download image"

```bash
# Check URL is accessible
curl -I https://example.com/image.png

# Try with different URL
zig build run -- https://httpbin.org/image/png brightness 20 -o output.png

# Check network connectivity
ping example.com
```

### Timeout Issues

Remote image downloads may timeout for large files. Consider downloading first:

```bash
# Download then process
curl -O https://example.com/large-image.jpg
zig build run -- large-image.jpg resize 2000 1500 -o processed.jpg
```

## Getting Help

### Debug Information

Always include this information when reporting issues:

```bash
# System information
uname -a
zig version

# Command that fails
zig build run -- image.png brightness 20 -o output.png

# Error output (full)
# Image details
file image.png
ls -la image.png
```

### Test Cases

Create minimal test cases:

```bash
# Create small test image
zig build run -- small.png brightness 20 -o test.png

# Test with different parameters
zig build run -- test.png contrast 1.5 -o test2.png
```

## Next Steps

- [Installation](installation.md) - Verify your setup
- [Examples](examples.md) - Working examples
- [Performance](performance.md) - Optimize your usage</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\troubleshooting.md