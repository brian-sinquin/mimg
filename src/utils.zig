const types = @import("types.zig");
const std = @import("std");
const img = @import("zigimg");
const http = std.http;

pub const output_path_buffer_size = 4096;

pub fn loadImage(ctx: *types.Context, path: []const u8) !img.Image {
    var read_buffer: [8192]u8 = undefined;
    return img.Image.fromFilePath(ctx.allocator, path, &read_buffer) catch |err| {
        std.log.err("Failed to load image '{s}': {}", .{ path, err });
        return err;
    };
}

pub fn saveImage(ctx: *types.Context, path: []const u8) !void {
    var buffer: [output_path_buffer_size]u8 = undefined;
    const resolved = try resolveOutputPath(ctx, path, &buffer);
    try saveImageToPath(ctx.image, ctx.allocator, resolved);
}

pub fn saveImageToPath(image: img.Image, allocator: std.mem.Allocator, resolved_path: []const u8) !void {
    var write_buffer: [img.io.DEFAULT_BUFFER_SIZE]u8 = undefined;
    try image.writeToFilePath(allocator, resolved_path, write_buffer[0..], .{ .png = .{} });
}

pub fn resolveOutputPath(ctx: *types.Context, filename: []const u8, buffer: []u8) ![]const u8 {
    if (ctx.output_directory) |dir| {
        try std.fs.cwd().makePath(dir);
        var fba = std.heap.FixedBufferAllocator.init(buffer);
        const joined = try std.fs.path.join(fba.allocator(), &[_][]const u8{ dir, filename });
        return @as([]const u8, joined);
    }
    return filename;
}

pub fn parseArgs(comptime type_list: []const type, it: *std.process.ArgIterator) !std.meta.Tuple(type_list) {
    var tuple: std.meta.Tuple(type_list) = undefined;
    inline for (type_list, 0..) |T, i| {
        const arg = it.next() orelse return types.ParseArgError.MissingArgument;
        if (T == []const u8) {
            tuple[i] = arg;
            continue;
        }

        switch (@typeInfo(T)) {
            .float => {
                tuple[i] = std.fmt.parseFloat(T, arg) catch {
                    return types.ParseArgError.InvalidArgument;
                };
            },
            .int => {
                tuple[i] = std.fmt.parseInt(T, arg, 10) catch {
                    return types.ParseArgError.InvalidArgument;
                };
            },
            else => @compileError("Unsupported parameter type in parseArgs"),
        }
    }
    return tuple;
}

pub fn parseNextArg(comptime T: type, it: *std.process.ArgIterator) types.ParseArgError!T {
    const arg = it.next() orelse {
        return types.ParseArgError.MissingArgument;
    };
    if (T == []const u8) {
        return arg;
    }

    return switch (@typeInfo(T)) {
        .float => std.fmt.parseFloat(T, arg) catch {
            return types.ParseArgError.InvalidArgument;
        },
        .int => std.fmt.parseInt(T, arg, 10) catch {
            return types.ParseArgError.InvalidArgument;
        },
        else => @compileError("Unsupported parameter type in parseNextArg"),
    };
}

/// Downloads the content of a file from a URL into memory (does not write to disk).
pub fn downloadFileToMemory(allocator: std.mem.Allocator, url: []const u8) ![]u8 {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    var allocating = std.Io.Writer.Allocating.init(allocator);
    defer allocating.deinit();

    const result = try client.fetch(.{
        .location = .{ .url = url },
        .response_writer = &allocating.writer,
    });

    if (result.status != .ok) return error.HttpError;

    return allocating.toOwnedSlice();
}
