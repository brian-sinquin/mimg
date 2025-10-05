const std = @import("std");
const gallery_data = @import("gallery_data.zig");

// Helper function to write example section
fn writeExampleSection(file: std.fs.File, example: gallery_data.GalleryExample) !void {
    var name_buf: [64]u8 = undefined;
    var desc_buf: [128]u8 = undefined;
    var img_buf: [512]u8 = undefined;

    const name_str = try std.fmt.bufPrint(&name_buf, "### {s}\n\n", .{example.name});
    try file.writeAll(name_str);

    const desc_str = try std.fmt.bufPrint(&desc_buf, "{s}\n\n", .{example.description});
    try file.writeAll(desc_str);

    // Build filename directly in the image markdown
    var filename_part: [256]u8 = undefined;
    var filename_len: usize = 0;

    for (example.args, 0..) |arg, i| {
        if (i > 0) {
            filename_part[filename_len] = '_';
            filename_len += 1;
        }
        @memcpy(filename_part[filename_len .. filename_len + arg.len], arg);
        filename_len += arg.len;
    }
    @memcpy(filename_part[filename_len .. filename_len + 9], "_lena.png");
    filename_len += 9;

    const filename = filename_part[0..filename_len];
    const img_str = try std.fmt.bufPrint(&img_buf, "![{s}](output/{s})\n\n", .{ example.name, filename });
    try file.writeAll(img_str);
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const cwd = std.fs.cwd();

    // Open gallery.md for writing
    const file = try cwd.createFile("examples/gallery/gallery.md", .{});
    defer file.close();

    // Write header
    try file.writeAll("# MIMG Image Processing Examples\n\n");
    try file.writeAll("This gallery showcases various image processing modifiers available in mimg.\n\n");
    try file.writeAll("All examples are generated from the original `lena.png` image.\n\n");
    try file.writeAll("*Special thanks to Lena Söderberg, whose iconic photograph has been a cornerstone of image processing research for decades.*\n\n");

    // Original image
    try file.writeAll("## Original Image\n\n");
    try file.writeAll("![Original](examples/lena.png)\n\n");

    // Individual modifiers
    try file.writeAll("## Individual Modifiers\n\n");

    for (gallery_data.individual_modifiers) |modifier| {
        try writeExampleSection(file, modifier);
    }

    // Combinations
    try file.writeAll("## Modifier Combinations\n\n");

    for (gallery_data.combinations) |combo| {
        try writeExampleSection(file, combo);
    }

    // Usage examples
    try file.writeAll("## Usage Examples\n\n");
    try file.writeAll("```bash\n");
    try file.writeAll("# Apply single modifier\n");
    try file.writeAll("mimg image.png sepia --output sepia_image.png\n\n");
    try file.writeAll("# Chain multiple modifiers\n");
    try file.writeAll("mimg image.png brightness 20 contrast 1.2 sharpen --output enhanced.png\n\n");
    try file.writeAll("# Generate this gallery\n");
    try file.writeAll("zig build gallery\n");
    try file.writeAll("```\n\n");

    try file.writeAll("## Available Modifiers\n\n");
    try file.writeAll("| Modifier | Description | Parameters |\n");
    try file.writeAll("|----------|-------------|------------|\n");
    try file.writeAll("| invert | Invert colors | None |\n");
    try file.writeAll("| grayscale | Convert to grayscale | None |\n");
    try file.writeAll("| brightness | Adjust brightness | value (-128 to 127) |\n");
    try file.writeAll("| contrast | Adjust contrast | factor (float) |\n");
    try file.writeAll("| saturation | Adjust saturation | factor (float) |\n");
    try file.writeAll("| gamma | Apply gamma correction | value (float) |\n");
    try file.writeAll("| sepia | Apply sepia tone | None |\n");
    try file.writeAll("| blur | Apply box blur | kernel_size (odd integer) |\n");
    try file.writeAll("| sharpen | Sharpen image | None |\n");
    try file.writeAll("| flip | Flip horizontally/vertically | direction |\n");
    try file.writeAll("| rotate | Rotate by angle | degrees (float) |\n");
    try file.writeAll("| resize | Resize image | width height |\n");
    try file.writeAll("| crop | Crop image | x y width height |\n");

    std.log.info("Generated examples/gallery/gallery.md gallery", .{});
}
