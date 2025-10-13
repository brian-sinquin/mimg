const types = @import("types.zig");
const std = @import("std");
const img = @import("zigimg");
const http = std.http;

pub const output_path_buffer_size = 4096;

pub fn loadImage(ctx: *types.Context, path: []const u8) !img.Image {
    var read_buffer: [8192]u8 = undefined;
    return img.Image.fromFilePath(ctx.allocator, path, &read_buffer) catch |err| {
        std.log.err("Failed to load image '{s}': {}", .{ path, err });
        return types.ImageError.LoadFailed;
    };
}

pub fn saveImage(ctx: *types.Context, path: []const u8) !void {
    var buffer: [output_path_buffer_size]u8 = undefined;
    const resolved = try resolveOutputPath(ctx, path, &buffer);
    try saveImageToPath(ctx, resolved);
}

pub fn saveImageToPath(ctx: *types.Context, resolved_path: []const u8) !void {
    // Ensure the output directory exists
    if (std.fs.path.dirname(resolved_path)) |dir| {
        std.fs.cwd().makePath(dir) catch |err| {
            std.log.err("Failed to create output directory '{s}': {}", .{ dir, err });
            return types.FileSystemError.DirectoryNotFound;
        };
    }

    var write_buffer: [img.io.DEFAULT_BUFFER_SIZE]u8 = undefined;

    // Determine format from file extension
    const ext = std.fs.path.extension(resolved_path);
    const format_options = getFormatOptionsFromExtension(ext);

    try ctx.image.writeToFilePath(ctx.allocator, resolved_path, write_buffer[0..], format_options);
}

pub fn joinPathWithDir(dir: ?[]const u8, filename: []const u8, buffer: []u8) ![]const u8 {
    // Input validation
    if (dir) |d| {
        if (d.len == 0) return error.InvalidDirectory;
        if (filename.len == 0) return error.InvalidFilename;
        if (d.len + filename.len + 1 > buffer.len) return error.PathTooLong;
    }

    if (dir) |d| {
        std.fs.cwd().makePath(d) catch {}; // Ignore error if already exists
        var fba = std.heap.FixedBufferAllocator.init(buffer);
        const joined = try std.fs.path.join(fba.allocator(), &[_][]const u8{ d, filename });
        return @as([]const u8, joined);
    }
    return filename;
}

pub fn resolveOutputPath(ctx: *types.Context, filename: []const u8, buffer: []u8) ![]const u8 {
    // Input validation
    if (filename.len == 0) return error.InvalidFilename;
    if (ctx.input_filename) |input| {
        if (input.len == 0) return error.InvalidInputFilename;
    }

    const input_stem = if (ctx.input_filename) |input|
        std.fs.path.stem(input)
    else
        std.fs.path.stem(filename);

    // Build the complete output path in one go to avoid buffer aliasing
    var fba = std.heap.FixedBufferAllocator.init(buffer);
    const allocator = fba.allocator();

    // Construct the base filename with modifiers
    const base_filename = if (std.mem.eql(u8, ctx.output_filename, "out")) blk: {
        // No modifiers, just use input stem
        break :blk try std.fmt.allocPrint(allocator, "{s}", .{input_stem});
    } else blk: {
        // Check if output_filename is a full filename (contains extension)
        if (std.fs.path.extension(ctx.output_filename).len > 0) {
            // Use as-is
            break :blk ctx.output_filename;
        } else {
            // Combine modifiers with input stem
            break :blk try std.fmt.allocPrint(allocator, "{s}_{s}", .{ ctx.output_filename, input_stem });
        }
    };

    // Add extension if needed
    const filename_with_ext = if (std.fs.path.extension(base_filename).len > 0)
        base_filename
    else blk: {
        break :blk try std.fmt.allocPrint(allocator, "{s}{s}", .{ base_filename, ctx.output_extension });
    };

    // Join with output directory if specified
    if (ctx.output_directory) |dir| {
        return std.fs.path.join(allocator, &[_][]const u8{ dir, filename_with_ext });
    } else {
        return filename_with_ext;
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
    // Input validation
    if (url.len == 0) return error.InvalidUrl;
    if (url.len > 2048) return error.UrlTooLong; // Reasonable URL length limit
    if (!isValidUrl(url)) return error.InvalidUrl;

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    var allocating = std.Io.Writer.Allocating.init(allocator);
    defer allocating.deinit();

    const result = try client.fetch(.{
        .location = .{ .url = url },
        .response_writer = &allocating.writer,
    });

    if (result.status != .ok) return types.NetworkError.HttpError;

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
    const supported = [_][]const u8{ ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".pbm", ".pgm", ".ppm", ".pcx", ".raw", ".qoi" };
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

pub fn expandWildcard(allocator: std.mem.Allocator, pattern: []const u8) ![]const []const u8 {
    // Find the last * in the pattern
    if (std.mem.lastIndexOf(u8, pattern, "*")) |star_pos| {
        // Split pattern into directory and filename pattern
        const dir_path = if (std.fs.path.dirname(pattern)) |dir| dir else ".";
        const filename_pattern = if (std.fs.path.dirname(pattern)) |dir|
            pattern[dir.len + 1 ..]
        else
            pattern;

        // Find * position in filename pattern
        const filename_star_pos = star_pos - if (std.fs.path.dirname(pattern)) |dir| dir.len + 1 else 0;

        // Split filename pattern into prefix*suffix
        const prefix = filename_pattern[0..filename_star_pos];
        const suffix = filename_pattern[filename_star_pos + 1 ..];

        var files = std.ArrayListUnmanaged([]const u8){};
        defer files.deinit(allocator);

        var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
        defer dir.close();

        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind == .file) {
                // Check if filename matches prefix*suffix pattern
                const matches = std.mem.startsWith(u8, entry.name, prefix) and
                    std.mem.endsWith(u8, entry.name, suffix);

                if (matches) {
                    // Create full path
                    const full_path = if (std.mem.eql(u8, dir_path, "."))
                        try allocator.dupe(u8, entry.name)
                    else
                        try std.fs.path.join(allocator, &[_][]const u8{ dir_path, entry.name });
                    try files.append(allocator, full_path);
                }
            }
        }

        return files.toOwnedSlice(allocator);
    }

    // No wildcard, return as-is
    const duped = try allocator.dupe(u8, pattern);
    return try allocator.dupe([]const u8, &[_][]const u8{duped});
}

pub fn loadImageFromSource(ctx: *types.Context, source: []const u8) !img.Image {
    if (isValidUrl(source)) {
        // Download from URL
        const image_data = try downloadFileToMemory(ctx.allocator, source);
        defer ctx.allocator.free(image_data);

        return img.Image.fromMemory(ctx.allocator, image_data) catch |url_err| {
            std.log.err("Failed to load image from URL '{s}': {}", .{ source, url_err });
            return types.ImageError.InvalidFormat;
        };
    } else {
        // Try to load as local file
        return loadImage(ctx, source);
    }
}

pub fn getPixel(pixels: []const img.color.Rgba32, width: usize, x: usize, y: usize) img.color.Rgba32 {
    // Input validation
    if (width == 0) unreachable;
    if (x >= width) unreachable;
    const idx = y * width + x;
    if (idx >= pixels.len) unreachable;
    return pixels[idx];
}

pub fn setPixel(pixels: []img.color.Rgba32, width: usize, x: usize, y: usize, color: img.color.Rgba32) void {
    // Input validation
    if (width == 0) unreachable;
    if (x >= width) unreachable;
    const idx = y * width + x;
    if (idx >= pixels.len) unreachable;
    pixels[idx] = color;
}

pub fn getPixelSafe(pixels: []const img.color.Rgba32, width: usize, height: usize, x: i32, y: i32) ?img.color.Rgba32 {
    if (x < 0 or y < 0 or x >= width or y >= height) return null;
    const idx = @as(usize, @intCast(y)) * width + @as(usize, @intCast(x));
    return pixels[idx];
}

pub fn getPixelClamped(pixels: []const img.color.Rgba32, width: usize, height: usize, x: i32, y: i32) img.color.Rgba32 {
    const clamped_x = std.math.clamp(x, 0, @as(i32, @intCast(width - 1)));
    const clamped_y = std.math.clamp(y, 0, @as(i32, @intCast(height - 1)));
    const idx = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x));
    return pixels[idx];
}

pub fn clampU8(value: f32) u8 {
    return @as(u8, @intFromFloat(std.math.clamp(value, 0.0, 255.0)));
}

pub fn clampI16ToU8(value: i16) u8 {
    return @as(u8, @intCast(std.math.clamp(value, 0, 255)));
}

pub fn rgbToLuminance(r: u8, g: u8, b: u8) f32 {
    return 0.299 * @as(f32, @floatFromInt(r)) +
        0.587 * @as(f32, @floatFromInt(g)) +
        0.114 * @as(f32, @floatFromInt(b));
}

pub fn rgbToLuminanceU8(r: u8, g: u8, b: u8) u8 {
    const lum = rgbToLuminance(r, g, b);
    return @as(u8, @intFromFloat(std.math.clamp(lum, 0.0, 255.0)));
}

pub fn createTempBufferReuse(allocator: std.mem.Allocator, pixels: []img.color.Rgba32, existing_buffer: ?[]img.color.Rgba32) ![]img.color.Rgba32 {
    if (existing_buffer) |buf| {
        if (buf.len >= pixels.len) {
            @memcpy(buf[0..pixels.len], pixels);
            return buf[0..pixels.len];
        } else {
            allocator.free(buf);
        }
    }
    return createTempBufferFromPixels(allocator, pixels);
}

pub fn createTempBufferFromPixels(allocator: std.mem.Allocator, pixels: []const img.color.Rgba32) ![]img.color.Rgba32 {
    const temp_pixels = try allocator.alloc(img.color.Rgba32, pixels.len);
    @memcpy(temp_pixels, pixels);
    return temp_pixels;
}

pub fn applyKernel3x3(
    pixels: []img.color.Rgba32,
    temp_pixels: []const img.color.Rgba32,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    kernel: [3][3]f32,
) struct { r: f32, g: f32, b: f32 } {
    // Input validation
    if (width == 0 or height == 0) unreachable;
    if (x >= width or y >= height) unreachable;
    if (temp_pixels.len != pixels.len) unreachable;

    var sum_r: f32 = 0.0;
    var sum_g: f32 = 0.0;
    var sum_b: f32 = 0.0;

    for (0..3) |ky| {
        for (0..3) |kx| {
            const px = @as(i32, @intCast(x)) + @as(i32, @intCast(kx)) - 1;
            const py = @as(i32, @intCast(y)) + @as(i32, @intCast(ky)) - 1;

            const pixel = getPixelClamped(temp_pixels, width, height, px, py);
            const weight = kernel[ky][kx];

            sum_r += @as(f32, @floatFromInt(pixel.r)) * weight;
            sum_g += @as(f32, @floatFromInt(pixel.g)) * weight;
            sum_b += @as(f32, @floatFromInt(pixel.b)) * weight;
        }
    }

    return .{ .r = sum_r, .g = sum_g, .b = sum_b };
}

pub fn applyKernel3x3i32(
    pixels: []img.color.Rgba32,
    temp_pixels: []const img.color.Rgba32,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    kernel: [3][3]i32,
) struct { r: i32, g: i32, b: i32 } {
    // Input validation
    if (width == 0 or height == 0) unreachable;
    if (x >= width or y >= height) unreachable;
    if (temp_pixels.len != pixels.len) unreachable;

    var sum_r: i32 = 0;
    var sum_g: i32 = 0;
    var sum_b: i32 = 0;

    for (0..3) |ky| {
        for (0..3) |kx| {
            const px = @as(i32, @intCast(x)) + @as(i32, @intCast(kx)) - 1;
            const py = @as(i32, @intCast(y)) + @as(i32, @intCast(ky)) - 1;

            const pixel = getPixelClamped(temp_pixels, width, height, px, py);
            const weight = kernel[ky][kx];

            sum_r += @as(i32, @intCast(pixel.r)) * weight;
            sum_g += @as(i32, @intCast(pixel.g)) * weight;
            sum_b += @as(i32, @intCast(pixel.b)) * weight;
        }
    }

    return .{ .r = sum_r, .g = sum_g, .b = sum_b };
}

pub fn convertToRgba32(ctx: *types.Context) !void {
    // Validate image dimensions to prevent excessive memory usage
    const max_dimension = 65535; // Reasonable limit for most systems
    const max_pixels = 50_000_000; // ~200MB for RGBA32 images

    if (ctx.image.width == 0 or ctx.image.height == 0) {
        std.log.err("Invalid image dimensions: {}x{}", .{ ctx.image.width, ctx.image.height });
        return types.ImageError.InvalidDimensions;
    }

    if (ctx.image.width > max_dimension or ctx.image.height > max_dimension) {
        std.log.err("Image dimensions too large: {}x{} (max allowed: {}x{})", .{ ctx.image.width, ctx.image.height, max_dimension, max_dimension });
        return types.ImageError.InvalidDimensions;
    }

    const total_pixels = std.math.mul(usize, ctx.image.width, ctx.image.height) catch {
        std.log.err("Image dimensions would cause integer overflow: {}x{}", .{ ctx.image.width, ctx.image.height });
        return types.ImageError.InvalidDimensions;
    };

    if (total_pixels > max_pixels) {
        std.log.err("Image too large: {} pixels (max allowed: {})", .{ total_pixels, max_pixels });
        return types.ImageError.InvalidDimensions;
    }

    try ctx.image.convert(ctx.allocator, .rgba32);
}

pub fn logVerbose(ctx: *types.Context, comptime fmt: []const u8, args: anytype) void {
    if (ctx.verbose) {
        std.log.info(fmt, args);
    }
}

pub fn logMemoryUsage(ctx: *types.Context, operation: []const u8) void {
    if (!ctx.verbose) return;

    const image_memory = ctx.image.width * ctx.image.height * @sizeOf(img.color.Rgba32);
    const temp_memory = if (ctx.temp_buffer) |buf| buf.len * @sizeOf(img.color.Rgba32) else 0;

    std.log.info("{s}: Image {}x{} = {} MB, Temp buffer = {} MB, Total = {} MB", .{ operation, ctx.image.width, ctx.image.height, image_memory / (1024 * 1024), temp_memory / (1024 * 1024), (image_memory + temp_memory) / (1024 * 1024) });
}

pub fn getFormatOptionsFromExtension(ext: []const u8) img.Image.EncoderOptions {
    if (std.mem.eql(u8, ext, ".png")) {
        return .{ .png = .{} };
    } else if (std.mem.eql(u8, ext, ".bmp")) {
        return .{ .bmp = .{} };
    } else if (std.mem.eql(u8, ext, ".tga")) {
        return .{ .tga = .{} };
    } else if (std.mem.eql(u8, ext, ".qoi")) {
        return .{ .qoi = .{} };
    } else if (std.mem.eql(u8, ext, ".pam")) {
        return .{ .pam = .{} };
    } else if (std.mem.eql(u8, ext, ".pbm")) {
        return .{ .pbm = .{} };
    } else if (std.mem.eql(u8, ext, ".pgm")) {
        return .{ .pgm = .{} };
    } else if (std.mem.eql(u8, ext, ".ppm")) {
        return .{ .ppm = .{} };
    } else if (std.mem.eql(u8, ext, ".pcx")) {
        return .{ .pcx = .{} };
    } else {
        // Default to PNG for unsupported extensions
        return .{ .png = .{} };
    }
}

/// Simple progress bar for long operations
pub const ProgressBar = struct {
    total: usize,
    current: std.atomic.Value(usize),
    width: usize = 50,
    mutex: std.Thread.Mutex = .{},

    pub fn init(total: usize) ProgressBar {
        return .{
            .total = total,
            .current = std.atomic.Value(usize).init(0),
            .width = 50,
            .mutex = .{},
        };
    }

    pub fn increment(self: *ProgressBar) void {
        _ = self.current.fetchAdd(1, .monotonic);
    }

    pub fn update(self: *ProgressBar) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const current = self.current.load(.monotonic);
        const percentage = @as(f32, @floatFromInt(current)) / @as(f32, @floatFromInt(self.total));
        const filled = @as(usize, @intFromFloat(percentage * @as(f32, @floatFromInt(self.width))));
        const percentage_int = @as(usize, @intFromFloat(percentage * 100));

        std.debug.print("\r[", .{});
        for (0..self.width) |i| {
            if (i < filled) {
                std.debug.print("=", .{});
            } else if (i == filled) {
                std.debug.print(">", .{});
            } else {
                std.debug.print(" ", .{});
            }
        }
        std.debug.print("] {d}/{d} ({d}%)", .{ current, self.total, percentage_int });
        if (current >= self.total) {
            std.debug.print("\n", .{});
        }
    }
};
