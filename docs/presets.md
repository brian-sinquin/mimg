# Presets

Presets allow you to save and reuse complex chains of image processing modifiers. Instead of typing long command lines repeatedly, create a preset file and apply it with a single flag.

## Creating Presets

Create a plain text file with one modifier per line. Empty lines and lines starting with `#` are ignored.

### Basic Preset

Create `vintage.preset`:
```
# Vintage photo effect
sepia
vignette 0.4
contrast 1.1
brightness 10
```

### Advanced Preset

Create `professional.preset`:
```
# Professional color correction pipeline
exposure 0.3
contrast 1.2
saturation 1.1
vibrance 0.2
sharpen
```

### Preset File Format

- **One modifier per line**
- **Parameters separated by spaces**
- **Comments** start with `#`
- **Empty lines** are ignored
- **No quotes** needed around parameters

**Example preset file:**
```text
# HDR-like effect with multiple adjustments
exposure 0.5
contrast 1.3
vibrance 0.3
gamma 1.2

# Add some sharpening to finish
sharpen
```

## Using Presets

Apply presets with the `--preset` or `-p` flag.

### Single Image
```bash
# Apply vintage preset
zig build run -- input.png --preset vintage.preset -o output.png

# Using shorthand
zig build run -- input.png -p professional.preset -o output.png
```

### Batch Processing
```bash
# Process all PNG files in current directory
zig build run -- *.png --preset batch-process.preset -d output/

# Process with custom output directory
zig build run -- photos/*.jpg --preset color-correction.preset -d processed/
```

### Combining with Additional Modifiers

Presets can be combined with additional modifiers for fine-tuning:

```bash
# Start with preset, then add more adjustments
zig build run -- input.png --preset base-correction.preset gamma 1.1 -o output.png

# Chain multiple presets (not directly supported, but you can combine manually)
zig build run -- input.png --preset preset1.preset brightness 5 --preset preset2.preset -o output.png
```

## Preset Organization

### File Naming Convention

Use descriptive names for your presets:
- `vintage.preset` - Vintage photo look
- `bw-film.preset` - Black and white film simulation
- `hdr-boost.preset` - HDR-like enhancement
- `portrait-retouch.preset` - Portrait enhancement
- `web-optimize.preset` - Web image optimization

### Directory Structure

Organize presets in folders:
```
presets/
├── color/
│   ├── vintage.preset
│   ├── bw-film.preset
│   └── duotone.preset
├── effects/
│   ├── vignette.preset
│   ├── sharpen.preset
│   └── blur.preset
└── workflows/
    ├── social-media.preset
    ├── print-ready.preset
    └── web-optimized.preset
```

## Preset Examples

### Photography Workflows

**Portrait Enhancement** (`portrait.preset`):
```
# Skin smoothing and enhancement
median-filter 3
vibrance 0.2
contrast 1.1
brightness 5
sharpen
```

**Landscape Boost** (`landscape.preset`):
```
# Enhance colors and details
vibrance 0.4
contrast 1.2
saturation 1.1
sharpen
```

### Creative Effects

**Vintage Film** (`film-vintage.preset`):
```
# Classic film look
sepia
vignette 0.3
contrast 1.2
brightness -10
gaussian-blur 0.3
```

**High Contrast B&W** (`bw-contrast.preset`):
```
# High contrast black and white
grayscale
contrast 2.0
brightness 20
sharpen
```

### Technical Presets

**Web Optimization** (`web-optimize.preset`):
```
# Prepare images for web
resize 1920 1080
gaussian-blur 0.2
sharpen
```

**Print Preparation** (`print-ready.preset`):
```
# High quality for printing
resize 3000 2000
sharpen
contrast 1.1
```

## Advanced Preset Techniques

### Conditional Processing

Use comments to document different options:
```text
# Base correction - uncomment one exposure line
# exposure 0.2    # Bright scene
exposure 0.0      # Normal scene
# exposure -0.2   # Dark scene

contrast 1.1
saturation 1.05
```

### Parameter Templates

Create presets with placeholder comments:
```text
# Color boost preset
# Adjust the vibrance value (0.0-1.0) based on image
vibrance 0.3
contrast 1.1
saturation 1.1
```

### Multi-Purpose Presets

Combine different effects in one preset:
```text
# Social media optimization
resize 1080 1080
vibrance 0.2
contrast 1.1
sharpen
vignette 0.1
```

## Preset Management

### Listing Presets

Keep an index of your presets:

**presets/README.md:**
```markdown
# Available Presets

## Color Correction
- `color-correction.preset` - Professional color grading
- `vintage-film.preset` - Classic film look
- `bw-contrast.preset` - High contrast black and white

## Effects
- `vignette-soft.preset` - Subtle corner darkening
- `sharpen-subtle.preset` - Gentle sharpening
- `blur-dreamy.preset` - Soft focus effect

## Workflows
- `social-media.preset` - Instagram-ready images
- `web-optimize.preset` - Fast-loading web images
- `print-ready.preset` - High-res print preparation
```

### Version Control

Track presets in git for collaboration:
```bash
# Add presets to version control
git add presets/
git commit -m "Add professional photography presets"
```

### Sharing Presets

Share presets with others:
```bash
# Package presets for sharing
tar -czf my-presets.tar.gz presets/
```

## Troubleshooting

### Common Preset Issues

**"Modifier not found"**
- Check spelling in preset file
- Ensure modifier exists (run `mimg --list-modifiers`)

**"Invalid parameter"**
- Verify parameter ranges in preset
- Check parameter order and types

**"File not found"**
- Use absolute paths or ensure preset file exists
- Check file permissions

### Debugging Presets

Use verbose mode to see preset processing:
```bash
zig build run -- input.png --preset mypreset.preset --verbose -o output.png
```

This shows each modifier being applied and any processing details.

## Next Steps

- [Usage Examples](examples.md) - See presets in action
- [Performance](performance.md) - Optimize your preset chains
- [Troubleshooting](troubleshooting.md) - Fix common preset issues</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\presets.md