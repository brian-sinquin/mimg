# File Formats

## Supported Formats

**Input**: PNG, JPEG, BMP, TGA, QOI, PAM, PBM, PGM, PPM, PCX  
**Output**: PNG, TGA, QOI, PAM, PBM, PGM, PPM

## Usage

```bash
# Auto-detect format from extension
zig build run -- photo.jpg brightness 20 -o enhanced.png

# Convert format explicitly
zig build run -- photo.jpg --output-extension .png -o converted.png
```

</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\formats.md