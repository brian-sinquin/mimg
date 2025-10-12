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
    try saveImageToPath(ctx, resolved);
}

pub fn saveImageToPath(ctx: *types.Context, resolved_path: []const u8) !void {
    var write_buffer: [img.io.DEFAULT_BUFFER_SIZE]u8 = undefined;
    try ctx.image.writeToFilePath(ctx.allocator, resolved_path, write_buffer[0..], .{ .png = .{} });
}

pub fn joinPathWithDir(dir: ?[]const u8, filename: []const u8, buffer: []u8) ![]const u8 {
    if (dir) |d| {
        std.fs.cwd().makePath(d) catch {}; // Ignore error if already exists
        var fba = std.heap.FixedBufferAllocator.init(buffer);
        const joined = try std.fs.path.join(fba.allocator(), &[_][]const u8{ d, filename });
        return @as([]const u8, joined);
    }
    return filename;
}

pub fn resolveOutputPath(ctx: *types.Context, filename: []const u8, buffer: []u8) ![]const u8 {
    if (std.fs.path.extension(filename).len > 0) {
        // Filename already has extension, use as is
        return joinPathWithDir(ctx.output_directory, filename, buffer);
    } else {
        // No extension, append output_extension
        const full_filename = try std.fmt.bufPrint(buffer, "{s}{s}", .{ filename, ctx.output_extension });
        return joinPathWithDir(ctx.output_directory, full_filename, buffer);
    }
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

pub fn parseArgsFromSlice(comptime type_list: []const type, args: []const []const u8, arg_index: *usize) !std.meta.Tuple(type_list) {
    var tuple: std.meta.Tuple(type_list) = undefined;
    inline for (type_list, 0..) |T, i| {
        if (arg_index.* >= args.len) return types.ParseArgError.MissingArgument;
        const arg = args[arg_index.*];
        arg_index.* += 1;

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
            else => @compileError("Unsupported parameter type in parseArgsFromSlice"),
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

pub fn isValidUrl(url: []const u8) bool {
    return std.mem.startsWith(u8, url, "http://") or std.mem.startsWith(u8, url, "https://");
}

pub fn getExtensionFromSource(source: []const u8) []const u8 {
    const ext = if (isValidUrl(source)) blk: {
        // Parse URL to get path
        const uri = std.Uri.parse(source) catch return ".png";
        const path = switch (uri.path) {
            .raw => |r| r,
            .percent_encoded => |p| p,
        };
        // Remove query params
        const query_start = std.mem.indexOf(u8, path, "?") orelse path.len;
        const clean_path = path[0..query_start];
        break :blk std.fs.path.extension(clean_path);
    } else std.fs.path.extension(source);

    if (ext.len == 0) return ".png";

    // Check if supported for saving
    const supported = [_][]const u8{ ".png", ".bmp", ".tga", ".pbm", ".pgm", ".ppm", ".pcx", ".raw", ".qoi" };
    for (supported) |sup| {
        if (std.mem.eql(u8, ext, sup)) return ext;
    }
    return ".png";
}

pub fn parseOutputFilename(filename: []const u8) struct { base: []const u8, ext: []const u8 } {
    const ext = std.fs.path.extension(filename);
    if (ext.len > 0) {
        const base = std.fs.path.stem(filename);
        return .{ .base = base, .ext = ext };
    } else {
        return .{ .base = filename, .ext = "" };
    }
}

pub fn loadImageFromSource(ctx: *types.Context, source: []const u8) !img.Image {
    if (isValidUrl(source)) {
        // Download from URL
        const image_data = try downloadFileToMemory(ctx.allocator, source);
        defer ctx.allocator.free(image_data);

        return img.Image.fromMemory(ctx.allocator, image_data) catch |url_err| {
            std.log.err("Failed to load image from URL '{s}': {}", .{ source, url_err });
            return url_err;
        };
    } else {
        // Try to load as local file
        return loadImage(ctx, source);
    }
}
