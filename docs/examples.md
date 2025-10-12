# Usage Examples

This guide provides practical examples of using mimg for various image processing tasks. Examples progress from basic operations to complex workflows.

## Quick Start Examples

### Basic Operations
```bash
# Convert to grayscale
zig build run -- input.png grayscale -o gray.png

# Adjust brightness and contrast
zig build run -- input.png brightness 20 contrast 1.2 -o bright.png

# Resize image
zig build run -- input.png resize 800 600 -o resized.png

# Rotate 90 degrees clockwise
zig build run -- input.png rotate 90 -o rotated.png
```

### Batch Processing
```bash
# Process all PNG files in current directory
zig build run -- *.png brightness 10 -d output/

# Convert all JPG files to PNG with brightness adjustment
zig build run -- *.jpg brightness 15 --output-extension .png -d converted/
```

## Color Correction Examples

### Basic Color Adjustments
```bash
# Natural color boost
zig build run -- photo.jpg vibrance 0.3 contrast 1.1 saturation 1.05 -o natural.png

# Fix underexposed image
zig build run -- dark.jpg exposure 0.5 brightness 15 -o corrected.png

# Enhance sunset colors
zig build run -- sunset.jpg vibrance 0.4 saturation 1.2 -o vibrant.png
```

### Professional Color Grading
```bash
# Cinematic look
zig build run -- scene.png contrast 1.3 vibrance 0.2 gamma 1.1 -o cinematic.png

# Product photography
zig build run -- product.jpg brightness 5 contrast 1.1 saturation 1.1 -o polished.png

# Portrait enhancement
zig build run -- portrait.jpg vibrance 0.2 contrast 1.1 brightness 3 -o enhanced.png
```

## Creative Effects

### Classic Film Looks
```bash
# Vintage sepia
zig build run -- photo.jpg sepia vignette 0.4 contrast 1.1 -o vintage.png

# High contrast black and white
zig build run -- photo.jpg grayscale contrast 2.0 brightness 10 -o bw.png

# Cross-processed look
zig build run -- photo.jpg hue-shift 30 contrast 1.3 saturation 1.2 -o cross.png
```

### Artistic Effects
```bash
# Oil painting
zig build run -- photo.jpg oil-painting 3 vibrance 0.3 -o oil.png

# Posterized art
zig build run -- photo.jpg posterize 8 contrast 1.5 -o poster.png

# Solarized effect
zig build run -- photo.jpg solarize 128 contrast 1.2 -o solarized.png
```

### Duotone Effects
```bash
# Classic cyanotype
zig build run -- photo.jpg duotone 0 50 100 200 150 50 -o cyanotype.png

# Warm sepia tones
zig build run -- photo.jpg duotone 75 30 0 255 180 100 -o sepia-warm.png

# Cool blue tones
zig build run -- photo.jpg duotone 0 20 80 150 200 255 -o cool-blue.png
```

## Filter Chains

### Sharpening Workflows
```bash
# Subtle sharpening
zig build run -- photo.jpg gaussian-blur 0.3 sharpen -o sharp.png

# Portrait sharpening
zig build run -- portrait.jpg median-filter 3 sharpen -o portrait-sharp.png

# Landscape enhancement
zig build run -- landscape.jpg vibrance 0.2 sharpen contrast 1.1 -o enhanced.png
```

### Noise Reduction
```bash
# Light noise reduction
zig build run -- noisy.jpg median-filter 3 -o denoised.png

# Heavy noise with detail preservation
zig build run -- very-noisy.jpg gaussian-blur 0.5 sharpen -o clean.png
```

### Creative Blurring
```bash
# Soft focus portrait
zig build run -- portrait.jpg gaussian-blur 1.5 brightness 5 -o soft.png

# Dreamy effect
zig build run -- landscape.jpg gaussian-blur 2.0 vibrance 0.3 -o dreamy.png
```

## Geometric Transformations

### Resizing Workflows
```bash
# Web optimization
zig build run -- large.jpg resize 1920 1080 sharpen -o web.png

# Thumbnail creation
zig build run -- photos/*.jpg resize 300 300 -d thumbnails/

# Print preparation
zig build run -- photo.jpg resize 3000 2000 sharpen -o print.png
```

### Cropping Examples
```bash
# Center crop to square
zig build run -- landscape.jpg crop 200 0 600 600 -o square.png

# Remove borders
zig build run -- scanned.jpg crop 50 50 700 500 -o cropped.png

# Focus on subject
zig build run -- group.jpg crop 100 50 400 300 -o subject.png
```

### Rotation Workflows
```bash
# Fix orientation
zig build run -- rotated.jpg rotate 90 -o corrected.png

# Create variations
for angle in 90 180 270; do
  zig build run -- input.png rotate $angle -o rotated-${angle}.png
done
```

## Complex Pipelines

### Professional Photo Editing
```bash
# Complete portrait retouch
zig build run -- portrait.jpg \
  median-filter 3 \
  vibrance 0.2 \
  contrast 1.1 \
  brightness 5 \
  gaussian-blur 0.2 \
  sharpen \
  -o retouched.png
```

### Social Media Optimization
```bash
# Instagram-ready image
zig build run -- photo.jpg \
  resize 1080 1080 \
  vibrance 0.3 \
  contrast 1.1 \
  brightness 10 \
  vignette 0.1 \
  sharpen \
  -o instagram.png
```

### Print Preparation
```bash
# High-quality print image
zig build run -- photo.jpg \
  resize 3000 2000 \
  exposure 0.2 \
  contrast 1.2 \
  vibrance 0.1 \
  sharpen \
  -o print-ready.png
```

## Preset-Based Workflows

### Using Presets
```bash
# Apply saved preset
zig build run -- photo.jpg --preset vintage.preset -o styled.png

# Combine preset with additional adjustments
zig build run -- photo.jpg --preset color-correction.preset gamma 1.1 -o final.png

# Batch processing with presets
zig build run -- raw-photos/*.jpg --preset professional.preset -d processed/
```

### Creating Custom Workflows
```bash
# Save complex pipeline as preset
echo "# Professional workflow
exposure 0.3
contrast 1.2
saturation 1.1
vibrance 0.2
sharpen" > professional.preset

# Use the preset
zig build run -- input.jpg --preset professional.preset -o output.png
```

## Performance Optimization

### Fast Processing Tips
```bash
# Apply transforms before filters (faster)
zig build run -- large.jpg resize 1000 1000 gaussian-blur 1.0 -o fast.png

# Use smaller kernel sizes for speed
zig build run -- image.jpg median-filter 3 -o quick-denoise.png

# Process in batches for efficiency
zig build run -- *.jpg brightness 10 contrast 1.1 -d batch-output/
```

### Memory-Efficient Processing
```bash
# Resize large images before complex operations
zig build run -- huge.jpg resize 2000 1500 oil-painting 3 -o efficient.png

# Use tiled processing for very large images
zig build run -- massive.jpg --verbose median-filter 5 -o processed.png
```

## Error Handling Examples

### Common Issues and Solutions
```bash
# Fix parameter range errors
zig build run -- image.jpg contrast 1.5 -o fixed.png  # Valid range

# Handle image size limits
zig build run -- huge.jpg resize 4000 3000 -o sized.png  # Within limits

# Fix crop bounds
zig build run -- image.jpg crop 0 0 800 600 -o cropped.png  # Valid coordinates
```

## Scripting Examples

### Bash Automation
```bash
#!/bin/bash
# Batch process all images in a directory
for file in photos/*.jpg; do
  base=$(basename "$file" .jpg)
  zig build run -- "$file" --preset enhance.preset -o "output/${base}.png"
done
```

### PowerShell Automation
```powershell
# Windows batch processing
Get-ChildItem "photos\*.jpg" | ForEach-Object {
  $base = $_.BaseName
  & zig build run -- $_.FullName --preset enhance.preset -o "output\$base.png"
}
```

## Next Steps

- [Available Modifiers](modifiers.md) - Complete reference of all effects
- [Presets](presets.md) - Save and reuse processing chains
- [Performance](performance.md) - Optimize your workflows
- [Troubleshooting](troubleshooting.md) - Fix common issues</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\examples.md