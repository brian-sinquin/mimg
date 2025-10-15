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

    // Read template file using the same allocator we will free with (avoid allocator mismatch)
    const template_content = try cwd.readFileAlloc(allocator, "examples/gallery/gallery.html.temp", 64 * 1024);
    defer allocator.free(template_content);

    // Generate individual modifiers section
    var individual_modifiers = try std.ArrayList(u8).initCapacity(allocator, 8192);
    defer individual_modifiers.deinit(allocator);
    var combinations = try std.ArrayList(u8).initCapacity(allocator, 8192);
    defer combinations.deinit(allocator);

    for (gallery_data.individual_modifiers) |modifier| {
        const snippet = renderExampleSectionAlloc(modifier, allocator) catch |e| blk: {
            std.log.err("render example failed: {s}", .{@errorName(e)});
            break :blk null;
        };
        if (snippet) |s| {
            defer allocator.free(s);
            try individual_modifiers.appendSlice(allocator, s);
        }
    }

    for (gallery_data.combinations) |combo| {
        const snippet = renderExampleSectionAlloc(combo, allocator) catch |e| blk: {
            std.log.err("render combo failed: {s}", .{@errorName(e)});
            break :blk null;
        };
        if (snippet) |s| {
            defer allocator.free(s);
            try combinations.appendSlice(allocator, s);
        }
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

    // Write final gallery.html (truncate to avoid stale bytes)
    const file = try cwd.createFile("examples/gallery/gallery.html", .{ .truncate = true });
    defer file.close();

    try file.writeAll(output.items);

    std.log.info("Generated examples/gallery/gallery.html gallery", .{});
}

// Helper function to write example section
fn renderExampleSectionAlloc(example: gallery_data.GalleryExample, allocator: std.mem.Allocator) ![]u8 {
    // Build filename using utility function
    const filename = try generateGalleryFilename(allocator, example.args);
    defer allocator.free(filename);

    // Build command string from args
    var command_str = try std.ArrayList(u8).initCapacity(allocator, 256);
    defer command_str.deinit(allocator);

    try command_str.appendSlice(allocator, "mimg lena.png ");
    for (example.args, 0..) |arg, i| {
        if (i > 0) {
            try command_str.append(allocator, ' ');
        }
        try command_str.appendSlice(allocator, arg);
    }

    // Allocate exact HTML snippet
    return try std.fmt.allocPrint(allocator, "<div class=\"image-item\">\n" ++
        "    <div class=\"command-ribbon\">\n" ++
        "        <span class=\"command-text\">{s}</span>\n" ++
        "        <button class=\"copy-button\" onclick=\"copyCommand('{s}', this)\">\n" ++
        "            <svg class=\"copy-icon\" viewBox=\"0 0 24 24\"><path d=\"M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z\"/></svg>\n" ++
        "            Copy\n" ++
        "        </button>\n" ++
        "    </div>\n" ++
        "    <img src=\"output/{s}\" alt=\"{s}\">\n" ++
        "    <div class=\"image-info\">\n" ++
        "        <div class=\"image-title\">{s}</div>\n" ++
        "        <div class=\"image-description\">{s}</div>\n" ++
        "    </div>\n" ++
        "</div>\n", .{ command_str.items, command_str.items, filename, example.name, example.name, example.description });
}
