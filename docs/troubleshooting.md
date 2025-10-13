# Troubleshooting

## Common Issues

**Invalid parameter range**: Check values are within limits  
**Kernel size must be odd**: Use 3, 5, 7 for filters  
**Crop out of bounds**: Ensure coordinates fit image  

## File Formats

**Failed to load**: Check file exists and is readable  
**Unsupported format**: Convert to PNG/JPG first  
Supported: PNG, JPG, BMP, TGA, QOI, PAM, PBM, PGM, PPM, PCX

## Memory/Performance  

**Out of memory**: Resize large images first  
**Slow processing**: Use smaller filter sizes, resize before complex operations  

## Build Issues

**Zig version**: Requires 0.15.1+  
**Clean build**: `rm -rf .zig-cache/ && zig build`</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\troubleshooting.md