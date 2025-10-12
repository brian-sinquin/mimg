const std = @import("std");
const gallery_data = @import("gallery_data.zig");

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
        try writeExampleSection(individual_modifiers.writer(allocator), modifier);
    }

    // Generate combinations section
    var combinations = try std.ArrayList(u8).initCapacity(allocator, 4096);
    defer combinations.deinit(allocator);

    for (gallery_data.combinations) |combo| {
        try writeExampleSection(combinations.writer(allocator), combo);
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
fn writeExampleSection(writer: anytype, example: gallery_data.GalleryExample) !void {
    var name_buf: [64]u8 = undefined;
    var desc_buf: [128]u8 = undefined;
    var img_buf: [512]u8 = undefined;

    const name_str = try std.fmt.bufPrint(&name_buf, "### {s}\n\n", .{example.name});
    try writer.writeAll(name_str);

    const desc_str = try std.fmt.bufPrint(&desc_buf, "{s}\n\n", .{example.description});
    try writer.writeAll(desc_str);

    // Build filename directly in the image markdown
    var filename_part: [256]u8 = undefined;
    var filename_len: usize = 0;

    for (example.args, 0..) |arg, i| {
        if (i > 0) {
            filename_part[filename_len] = '_';
            filename_len += 1;
        }

        // Sanitize the argument by replacing dots with underscores
        for (arg) |char| {
            if (char == '.') {
                filename_part[filename_len] = '_';
            } else {
                filename_part[filename_len] = char;
            }
            filename_len += 1;
        }
    }
    @memcpy(filename_part[filename_len .. filename_len + 9], "_lena.png");
    filename_len += 9;

    const filename = filename_part[0..filename_len];
    const img_str = try std.fmt.bufPrint(&img_buf, "![{s}](output/{s})\n\n", .{ example.name, filename });
    try writer.writeAll(img_str);
}
