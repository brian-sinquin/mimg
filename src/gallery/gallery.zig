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

        // Sanitize the argument by replacing dots with underscores and '#' with '0x'
        for (arg) |char| {
            if (char == '.') {
                try writer.writeByte('_');
            } else if (char == '#') {
                try writer.writeAll("0x");
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
    const template_content = try cwd.readFileAlloc(std.heap.page_allocator, "examples/gallery/gallery.html.temp", 10 * 1024);
    defer allocator.free(template_content);

    // Generate individual modifiers section
    var individual_modifiers = try std.ArrayList(u8).initCapacity(allocator, 8192);
    defer individual_modifiers.deinit(allocator);
    var combinations = try std.ArrayList(u8).initCapacity(allocator, 8192);
    defer combinations.deinit(allocator);

    for (gallery_data.individual_modifiers) |modifier| {
        var temp: [4096]u8 = [_]u8{0} ** 4096;
        const written = writeExampleSectionFast(temp[0..], modifier, allocator) catch 0;
        try individual_modifiers.appendSlice(allocator, temp[0..written]);
    }

    for (gallery_data.combinations) |combo| {
        var temp: [4096]u8 = [_]u8{0} ** 4096;
        const written = writeExampleSectionFast(temp[0..], combo, allocator) catch 0;
        try combinations.appendSlice(allocator, temp[0..written]);
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

    // Write final gallery.html
    const file = try cwd.createFile("examples/gallery/gallery.html", .{});
    defer file.close();

    try file.writeAll(output.items);

    std.log.info("Generated examples/gallery/gallery.html gallery", .{});
}

// Helper function to write example section
fn writeExampleSectionFast(buf: []u8, example: gallery_data.GalleryExample, allocator: std.mem.Allocator) !usize {
    // Build filename using utility function
    const filename = try generateGalleryFilename(allocator, example.args);
    defer allocator.free(filename);

    const rest = std.fmt.bufPrint(buf, "<div class=\"image-item\">\n    <img src=\"output/{s}\" alt=\"{s}\">\n    <div class=\"image-info\">\n        <div class=\"image-title\">{s}</div>\n        <div class=\"image-description\">{s}</div>\n    </div>\n</div>\n", .{ filename, example.name, example.name, example.description }) catch return 0;
    return buf.len - rest.len;
}
