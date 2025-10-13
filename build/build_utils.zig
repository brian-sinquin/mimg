const std = @import("std");

/// Generate a sanitized filename from command line arguments for gallery examples
pub fn generateGalleryFilename(allocator: std.mem.Allocator, args: []const []const u8) ![]const u8 {
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
