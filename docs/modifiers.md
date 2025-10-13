# Available Modifiers

## Options

| Option | Description |
|--------|-------------|
| `-o <file>` | Output filename |
| `-d <dir>` | Output directory |
| `--preset <file>` | Load modifier chain from preset file |
| `--help` | Show help message |

## Modifiers

| Modifier | Usage | Description |
|----------|-------|-------------|
| `brightness` | `brightness <value>` | Adjust brightness (-255 to 255) |
| `contrast` | `contrast <factor>` | Adjust contrast (0.0 to 3.0) |
| `saturation` | `saturation <factor>` | Adjust color saturation |
| `grayscale` | `grayscale` | Convert to grayscale |
| `sepia` | `sepia` | Apply sepia tone |
| `invert` | `invert` | Invert colors |
| `blur` | `blur <size>` | Box blur (kernel size) |
| `gaussian-blur` | `gaussian-blur <sigma>` | Gaussian blur |
| `sharpen` | `sharpen` | Sharpen image |
| `edge-detect` | `edge-detect` | Detect edges |
| `emboss` | `emboss <strength>` | Emboss effect |
| `oil-painting` | `oil-painting <radius>` | Oil painting effect |
| `posterize` | `posterize <levels>` | Reduce colors |
| `pixelate` | `pixelate <size>` | Pixelation effect |
| `resize` | `resize <width> <height>` | Resize image |
| `crop` | `crop <x> <y> <w> <h>` | Crop image |
| `rotate` | `rotate <degrees>` | Rotate image |
| `flip` | `flip <horizontal\|vertical>` | Flip image |

## Usage

```bash
# Single modifier
zig build run -- input.png brightness 20 -o output.png

# Chain multiple modifiers
zig build run -- input.png brightness 10 contrast 1.2 sharpen -o output.png
```</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\modifiers.md