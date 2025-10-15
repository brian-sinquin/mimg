# Presets

Save and reuse modifier chains using preset files.

## Creating Presets

Create a text file with one modifier per line:

`vintage.preset`:
```
sepia
vignette 0.4
contrast 1.1
```

## Using Presets

```bash
# Apply preset
zig build run -- input.png --preset vintage.preset -o output.png

# Batch processing
zig build run -- *.jpg --preset enhance.preset -d output/
```

## Examples

`portrait.preset`:
```
median-filter 3
vibrance 0.2
contrast 1.1
sharpen
```

`vintage.preset`:
```
sepia
vignette 0.3
contrast 1.2
```</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\presets.md