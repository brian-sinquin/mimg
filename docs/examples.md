# Examples

## Basic Operations

```bash
# Convert to grayscale
zig build run -- input.png grayscale -o gray.png

# Adjust brightness and contrast
zig build run -- input.png brightness 20 contrast 1.2 -o bright.png

# Resize image
zig build run -- input.png resize 800 600 -o resized.png

# Rotate 90 degrees
zig build run -- input.png rotate 90 -o rotated.png
```

## Color Adjustments

```bash
# Natural enhancement
zig build run -- photo.jpg vibrance 0.3 contrast 1.1 -o enhanced.png

# Fix underexposed image
zig build run -- dark.jpg exposure 0.5 brightness 15 -o corrected.png

# Vintage sepia effect
zig build run -- photo.jpg sepia vignette 0.4 -o vintage.png
```

## Creative Effects

```bash
# Oil painting effect
zig build run -- photo.jpg oil-painting 3 vibrance 0.3 -o oil.png

# Posterized art
zig build run -- photo.jpg posterize 8 contrast 1.5 -o poster.png

# Edge detection
zig build run -- photo.jpg edge-detect posterize 6 -o comic.png
```

## Chaining Operations

```bash
# Professional portrait
zig build run -- portrait.jpg \
  median-filter 3 \
  vibrance 0.2 \
  contrast 1.1 \
  sharpen \
  -o enhanced.png

# Batch processing
zig build run -- *.jpg brightness 10 contrast 1.1 -d output/
```</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\examples.md