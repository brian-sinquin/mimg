const std = @import("std");
const gallery_data = @import("gallery_data.zig");

// Helper function to generate gallery filename
fn generateGalleryFilename(allocator: std.mem.Allocator, args: []const []const u8) ![]const u8 {
    var filename_parts = try std.ArrayList(u8).initCapacity(allocator, 256);
    defer filename_parts.deinit(allocator);

    var writer = filename_parts.writer(allocator);

    for (args, 0..) |arg, i| {
        if (i > 0) {
            try writer.writeByte('_');
        }

        // Sanitize the argument by replacing dots with underscores
        for (arg) |char| {
            if (char == '.') {
                try writer.writeByte('_');
            } else {
                try writer.writeByte(char);
            }
        }
    }

    try writer.writeAll("_lena.png");
    return try filename_parts.toOwnedSlice(allocator);
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const cwd = std.fs.cwd();

    // Read template file
    const template_content = try cwd.readFileAlloc(allocator, "examples/gallery/gallery.md.temp", 10 * 1024);
    defer allocator.free(template_content);

    // Generate individual modifiers section
    var individual_modifiers = try std.ArrayList(u8).initCapacity(allocator, 4096);
    defer individual_modifiers.deinit(allocator);

    for (gallery_data.individual_modifiers) |modifier| {
        try writeExampleSection(individual_modifiers.writer(allocator), modifier, allocator);
    }

    // Generate combinations section
    var combinations = try std.ArrayList(u8).initCapacity(allocator, 4096);
    defer combinations.deinit(allocator);

    for (gallery_data.combinations) |combo| {
        try writeExampleSection(combinations.writer(allocator), combo, allocator);
    }

    // Replace placeholders in template
    var output = try std.ArrayList(u8).initCapacity(allocator, 8192);
    defer output.deinit(allocator);

    var lines_iter = std.mem.splitSequence(u8, template_content, "\n");
    while (lines_iter.next()) |line| {
        if (std.mem.indexOf(u8, line, "{{INDIVIDUAL_MODIFIERS}}")) |_| {
            try output.appendSlice(allocator, individual_modifiers.items);
        } else if (std.mem.indexOf(u8, line, "{{MODIFIER_COMBINATIONS}}")) |_| {
            try output.appendSlice(allocator, combinations.items);
        } else {
            try output.appendSlice(allocator, line);
            try output.append(allocator, '\n');
        }
    }

    // Write final gallery.md
    const file = try cwd.createFile("examples/gallery/gallery.md", .{});
    defer file.close();

    try file.writeAll(output.items);

    std.log.info("Generated examples/gallery/gallery.md gallery", .{});
}

// Helper function to write example section
fn writeExampleSection(writer: anytype, example: gallery_data.GalleryExample, allocator: std.mem.Allocator) !void {
    // Build filename using utility function
    const filename = try generateGalleryFilename(allocator, example.args);
    defer allocator.free(filename);

    try writer.print(
        \\<div class="image-item">
        \\    <img src="output/{s}" alt="{s}">
        \\    <div class="image-info">
        \\        <div class="image-title">{s}</div>
        \\        <div class="image-description">{s}</div>
        \\    </div>
        \\</div>
        \\
    , .{ filename, example.name, example.name, example.description });
}
