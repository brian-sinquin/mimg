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

// parseNextArg removed (unused)

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

// getPixel/setPixel removed (not used)

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

// Inline helpers for better performance - these are called frequently in hot paths
pub inline fn clampU8(value: f32) u8 {
    return @as(u8, @intFromFloat(std.math.clamp(value, 0.0, 255.0)));
}

pub inline fn u8ToF32(value: u8) f32 {
    return @as(f32, @floatFromInt(value));
}

pub inline fn luminanceF32(r: f32, g: f32, b: f32) f32 {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

/// Comptime constants for common mathematical operations (calculated once at compile time)
pub const INV_255: f32 = 1.0 / 255.0;
pub const INV_3: f32 = 1.0 / 3.0;
pub const INV_2: f32 = 1.0 / 2.0;
pub const SQRT_3: f32 = std.math.sqrt(3.0);
pub const DEG_TO_RAD: f32 = std.math.pi / 180.0;

/// Normalize u8 to 0.0-1.0 range (faster than division with multiplication by reciprocal)
pub inline fn normalizeU8(value: u8) f32 {
    return u8ToF32(value) * INV_255;
}

// clampI16ToU8 removed (unused)

pub fn rgbToLuminance(r: u8, g: u8, b: u8) f32 {
    return luminanceF32(u8ToF32(r), u8ToF32(g), u8ToF32(b));
}

pub fn rgbToLuminanceU8(r: u8, g: u8, b: u8) u8 {
    return clampU8(rgbToLuminance(r, g, b));
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

/// Parse a hex color string into an Rgba32.
/// Supports "#RRGGBB" and "#RRGGBBAA" (case-insensitive). Returns error.InvalidHexColor on bad input.
pub fn parseHexColor(hex: []const u8) !img.color.Rgba32 {
    if (hex.len != 7 and hex.len != 9) {
        return error.InvalidHexColor;
    }
    if (hex[0] != '#') {
        return error.InvalidHexColor;
    }

    // Parse RGB components
    const r = try std.fmt.parseInt(u8, hex[1..3], 16);
    const g = try std.fmt.parseInt(u8, hex[3..5], 16);
    const b = try std.fmt.parseInt(u8, hex[5..7], 16);

    // Parse alpha if provided, otherwise default to 255
    const a = if (hex.len == 9) try std.fmt.parseInt(u8, hex[7..9], 16) else @as(u8, 255);

    return img.color.Rgba32{ .r = r, .g = g, .b = b, .a = a };
}

/// Generate a sanitized filename from command line arguments for gallery examples
pub fn parseArgsFromSlice(comptime param_types: []const type, args: []const []const u8, arg_index: *usize) !std.meta.Tuple(param_types) {
    var result: std.meta.Tuple(param_types) = undefined;
    inline for (param_types, 0..) |param_type, i| {
        if (arg_index.* >= args.len) {
            return types.ParseArgError.MissingArgument;
        }
        const arg_str = args[arg_index.*];
        arg_index.* += 1;

        result[i] = try parseArg(param_type, arg_str);
    }
    return result;
}

fn parseArg(comptime T: type, arg: []const u8) !T {
    return switch (T) {
        []const u8 => arg,
        u8 => std.fmt.parseInt(u8, arg, 10) catch return types.ParseArgError.InvalidArgument,
        u16 => std.fmt.parseInt(u16, arg, 10) catch return types.ParseArgError.InvalidArgument,
        u32 => std.fmt.parseInt(u32, arg, 10) catch return types.ParseArgError.InvalidArgument,
        usize => std.fmt.parseInt(usize, arg, 10) catch return types.ParseArgError.InvalidArgument,
        i8 => std.fmt.parseInt(i8, arg, 10) catch return types.ParseArgError.InvalidArgument,
        i16 => std.fmt.parseInt(i16, arg, 10) catch return types.ParseArgError.InvalidArgument,
        i32 => std.fmt.parseInt(i32, arg, 10) catch return types.ParseArgError.InvalidArgument,
        isize => std.fmt.parseInt(isize, arg, 10) catch return types.ParseArgError.InvalidArgument,
        f32 => std.fmt.parseFloat(f32, arg) catch return types.ParseArgError.InvalidArgument,
        f64 => std.fmt.parseFloat(f64, arg) catch return types.ParseArgError.InvalidArgument,
        else => @compileError("Unsupported parameter type: " ++ @typeName(T)),
    };
}
