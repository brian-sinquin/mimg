# Library API

mimg can be used as a Zig library for programmatic image processing. This guide covers the API for integrating mimg into your Zig applications.

## Setup

Add mimg as a dependency in your `build.zig.zon`:

```zig
.dependencies = .{
    .mimg = .{
        .url = "https://github.com/brian-sinquin/mimg/archive/main.tar.gz",
        .hash = "...", // Run `zig fetch` to get the hash
    },
},
```

In your `build.zig`:

```zig
const mimg_dep = b.dependency("mimg", .{});
exe.root_module.addImport("mimg", mimg_dep.module("mimg"));
```

## Basic Usage

### Loading and Processing Images

```zig
const std = @import("std");
const img = @import("zigimg");
const mimg = @import("mimg");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create processing context
    var ctx = mimg.types.Context.init(allocator);
    defer ctx.deinit();

    // Load image
    const image = try img.Image.fromFilePath(allocator, "input.png");
    defer image.deinit(allocator);
    ctx.setImage(image);

    // Apply processing
    try mimg.basic.adjustBrightness(&ctx, .{20});
    try mimg.basic.adjustContrast(&ctx, .{1.2});

    // Save result
    try ctx.image.writeToFilePath("output.png", .{});
}
```

## Context Management

The `Context` struct manages image state and temporary buffers:

```zig
var ctx = mimg.types.Context.init(allocator);
defer ctx.deinit();

// Context automatically:
// - Manages image buffer
// - Reuses temporary buffers for filters
// - Handles memory allocation/deallocation
// - Provides error handling
```

### Context Methods

```zig
// Set the image to process
ctx.setImage(loaded_image);

// Get current image dimensions
const width = ctx.image.width;
const height = ctx.image.height;

// Check if image is loaded
if (ctx.image.pixels != null) {
    // Image is ready for processing
}
```

## Available Functions

All processing functions follow the pattern:
```zig
functionName(context: *Context, args: anytype) !void
```

### Color Adjustments

```zig
// Brightness adjustment (-255 to 255)
try mimg.basic.adjustBrightness(&ctx, .{50});

// Contrast adjustment (0.0 to 3.0)
try mimg.basic.adjustContrast(&ctx, .{1.5});

// Saturation adjustment (0.0 to 3.0)
try mimg.basic.adjustSaturation(&ctx, .{0.8});

// Gamma correction (0.1 to 3.0)
try mimg.basic.adjustGamma(&ctx, .{1.2});

// Hue rotation (0 to 360 degrees)
try mimg.basic.hueShiftImage(&ctx, .{45});

// Exposure adjustment (-2.0 to 2.0 stops)
try mimg.basic.adjustExposure(&ctx, .{0.5});

// Vibrance enhancement (0.0 to 1.0)
try mimg.basic.adjustVibrance(&ctx, .{0.3});

// Histogram equalization
try mimg.basic.equalizeImage(&ctx, .{});
```

### Color Effects

```zig
// Convert to grayscale
try mimg.basic.grayscaleImage(&ctx, .{});

// Invert all colors
try mimg.basic.invertColors(&ctx, .{});

// Apply sepia tone
try mimg.basic.applySepia(&ctx, .{});

// Binary threshold (0 to 255)
try mimg.basic.thresholdImage(&ctx, .{128});

// Solarization effect (0 to 255)
try mimg.basic.solarizeImage(&ctx, .{180});

// Posterize levels (2 to 16)
try mimg.basic.posterizeImage(&ctx, .{8});

// Color tint (r, g, b, strength)
try mimg.basic.colorizeImage(&ctx, .{255, 128, 0, 0.3});

// Duotone gradient (shadow_r, shadow_g, shadow_b, highlight_r, highlight_g, highlight_b)
try mimg.basic.duotoneImage(&ctx, .{0, 50, 100, 255, 220, 150});
```

### Filters

```zig
// Gaussian blur (sigma 0.5 to 5.0)
try mimg.filters.gaussianBlurImage(&ctx, .{1.5});

// Box blur (kernel size 3, 5, 7, 9...)
try mimg.filters.boxBlurImage(&ctx, .{5});

// Sharpen image
try mimg.filters.sharpenImage(&ctx, .{});

// Edge detection (Sobel)
try mimg.filters.edgeDetectImage(&ctx, .{});

// Emboss effect
try mimg.filters.embossImage(&ctx, .{});

// Median filter for noise (kernel size 3, 5, 7)
try mimg.filters.medianFilterImage(&ctx, .{3});

// Add noise (0.0 to 1.0)
try mimg.filters.noiseImage(&ctx, .{0.1});

// Vignette effect (0.0 to 1.0)
try mimg.filters.vignetteImage(&ctx, .{0.5});

// Pixelation (2 to 50)
try mimg.filters.pixelateImage(&ctx, .{8});

// Oil painting effect (1 to 5)
try mimg.filters.oilPaintingImage(&ctx, .{3});
```

### Geometric Transforms

```zig
// Resize image (1 to 65535)
try mimg.transforms.resizeImage(&ctx, .{1024, 768});

// Crop rectangle (x, y, width, height)
try mimg.transforms.cropImage(&ctx, .{100, 50, 800, 600});

// Rotate (90, 180, 270 degrees)
try mimg.transforms.rotateImage(&ctx, .{90});

// Flip horizontally
try mimg.transforms.flipImage(&ctx, .{"horizontal"});

// Flip vertically
try mimg.transforms.flipImage(&ctx, .{"vertical"});
```

## Advanced Usage

### Chaining Operations

```zig
// Chain multiple operations
try mimg.basic.adjustBrightness(&ctx, .{30});
try mimg.basic.adjustContrast(&ctx, .{1.2});
try mimg.filters.gaussianBlurImage(&ctx, .{0.8});
try mimg.filters.sharpenImage(&ctx, .{});
```

### Batch Processing

```zig
const std = @import("std");
const img = @import("zigimg");
const mimg = @import("mimg");

pub fn processBatch(allocator: std.mem.Allocator, file_paths: []const []const u8) !void {
    var ctx = mimg.types.Context.init(allocator);
    defer ctx.deinit();

    for (file_paths) |file_path| {
        // Load image
        const image = try img.Image.fromFilePath(allocator, file_path);
        defer image.deinit(allocator);
        ctx.setImage(image);

        // Apply processing
        try mimg.basic.adjustBrightness(&ctx, .{20});
        try mimg.basic.adjustContrast(&ctx, .{1.1});

        // Generate output path
        const output_path = try std.fmt.allocPrint(allocator, "output/{s}", .{std.fs.path.basename(file_path)});
        defer allocator.free(output_path);

        // Save result
        try ctx.image.writeToFilePath(output_path, .{});
    }
}
```

### Custom Processing Pipeline

```zig
const ProcessingPipeline = struct {
    ctx: mimg.types.Context,

    pub fn init(allocator: std.mem.Allocator) ProcessingPipeline {
        return .{ .ctx = mimg.types.Context.init(allocator) };
    }

    pub fn deinit(self: *ProcessingPipeline) void {
        self.ctx.deinit();
    }

    pub fn applyPreset(self: *ProcessingPipeline, preset_name: []const u8) !void {
        if (std.mem.eql(u8, preset_name, "vintage")) {
            try mimg.basic.applySepia(&self.ctx, .{});
            try mimg.filters.vignetteImage(&self.ctx, .{0.4});
            try mimg.basic.adjustContrast(&self.ctx, .{1.1});
        } else if (std.mem.eql(u8, preset_name, "bw")) {
            try mimg.basic.grayscaleImage(&self.ctx, .{});
            try mimg.basic.adjustContrast(&self.ctx, .{1.3});
        }
    }
};
```

## Error Handling

mimg uses Zig's error union types for robust error handling:

```zig
const result = mimg.basic.adjustBrightness(&ctx, .{300});
if (result) {
    std.debug.print("Success!\n", .{});
} else |err| switch (err) {
    error.InvalidParameters => std.debug.print("Brightness value out of range\n", .{}),
    error.OutOfMemory => std.debug.print("Not enough memory\n", .{}),
    error.ImageTooLarge => std.debug.print("Image exceeds size limits\n", .{}),
    else => std.debug.print("Processing error: {}\n", .{err}),
}
```

### Common Error Types

- `InvalidParameters` - Parameter values outside allowed ranges
- `OutOfMemory` - Insufficient memory for operation
- `ImageTooLarge` - Image exceeds 65535×65535 or 50M pixels
- `UnsupportedOperation` - Operation not supported for current image format
- `FileSystemError` - File I/O issues

## Memory Management

### Automatic Buffer Management

```zig
var ctx = mimg.types.Context.init(allocator);
defer ctx.deinit();

// Temporary buffers are automatically:
// - Allocated when needed
// - Reused across operations
// - Freed when context is deinitialized
```

### Memory-Efficient Processing

```zig
// For large images, consider processing in sections
const max_dimension = 2048;
if (ctx.image.width > max_dimension or ctx.image.height > max_dimension) {
    // Resize before applying memory-intensive filters
    try mimg.transforms.resizeImage(&ctx, .{max_dimension, max_dimension});
}

try mimg.filters.medianFilterImage(&ctx, .{5}); // Now uses less memory
```

## Type Definitions

### Context Structure

```zig
pub const Context = struct {
    allocator: std.mem.Allocator,
    image: img.Image,
    temp_buffer: ?[]u8, // Reused for filters

    pub fn init(allocator: std.mem.Allocator) Context
    pub fn deinit(self: *Context) void
    pub fn setImage(self: *Context, image: img.Image) void
};
```

### Function Signatures

All processing functions use this pattern:

```zig
pub fn functionName(ctx: *Context, args: anytype) !void
```

Where `args` is a tuple/struct containing the function parameters.

## Integration Examples

### Web Server Integration

```zig
// Process uploaded image
pub fn processUpload(allocator: std.mem.Allocator, image_data: []const u8) ![]u8 {
    var ctx = mimg.types.Context.init(allocator);
    defer ctx.deinit();

    // Load from memory
    const image = try img.Image.fromMemory(allocator, image_data);
    defer image.deinit(allocator);
    ctx.setImage(image);

    // Apply processing
    try mimg.basic.adjustBrightness(&ctx, .{20});
    try mimg.filters.gaussianBlurImage(&ctx, .{0.5});

    // Return processed image data
    return try ctx.image.toPng(allocator);
}
```

### Image Processing Library

```zig
pub const ImageProcessor = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ImageProcessor {
        return .{ .allocator = allocator };
    }

    pub fn enhancePhoto(self: ImageProcessor, image_path: []const u8) !void {
        var ctx = mimg.types.Context.init(self.allocator);
        defer ctx.deinit();

        const image = try img.Image.fromFilePath(self.allocator, image_path);
        defer image.deinit(self.allocator);
        ctx.setImage(image);

        // Professional enhancement pipeline
        try mimg.basic.adjustExposure(&self, .{0.3});
        try mimg.basic.adjustContrast(&self, .{1.1});
        try mimg.basic.adjustVibrance(&self, .{0.2});
        try mimg.filters.sharpenImage(&self, .{});

        const output_path = try std.fmt.allocPrint(self.allocator, "enhanced_{s}", .{std.fs.path.basename(image_path)});
        defer self.allocator.free(output_path);

        try ctx.image.writeToFilePath(output_path, .{});
    }
};
```

## Performance Considerations

### SIMD Optimization

mimg automatically uses SIMD instructions when available:

- Color operations use `@Vector(4, f32)` for 4-pixel parallel processing
- Integer operations use `@Vector(4, u8)` for byte-level parallelism
- Performance scales with CPU SIMD capabilities

### Memory Efficiency

- Buffer reuse minimizes allocations
- Tiled processing for large images (> 2048×2048)
- Automatic cleanup prevents memory leaks

### Benchmarking

```zig
// Time your operations
const start = std.time.nanoTimestamp();
// ... processing ...
const end = std.time.nanoTimestamp();
const duration_ms = @intToFloat(f64, end - start) / 1_000_000.0;
std.debug.print("Processing took: {d:.2}ms\n", .{duration_ms});
```

## Next Steps

- [Installation](installation.md) - Set up mimg for development
- [Examples](examples.md) - More integration examples
- [Troubleshooting](troubleshooting.md) - Debug library usage issues</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\library.md