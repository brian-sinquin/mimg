const img = @import("zigimg");
const std = @import("std");
const Context = @import("types.zig").Context;
const utils = @import("utils.zig");
const math = std.math;

pub fn resizeImage(ctx: *Context, args: anytype) !void {
    const new_width = args[0];
    const new_height = args[1];

    // Input validation
    if (new_width <= 0 or new_height <= 0) {
        return error.InvalidDimensions;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Resizing image to {}x{}", .{ new_width, new_height });
    utils.logMemoryUsage(ctx, "Resize start");

    const pixels = ctx.image.pixels.rgba32;
    const old_width = ctx.image.width;
    const old_height = ctx.image.height;

    // Create new image with target dimensions
    const new_pixels = try ctx.allocator.alloc(img.color.Rgba32, new_width * new_height);

    // Initialize all pixels to transparent
    @memset(new_pixels, img.color.Rgba32{ .r = 0, .g = 0, .b = 0, .a = 0 });

    // Nearest neighbor scaling
    for (0..new_height) |y| {
        const src_y = @as(usize, @intFromFloat(@as(f32, @floatFromInt(y)) * @as(f32, @floatFromInt(old_height)) / @as(f32, @floatFromInt(new_height))));
        const src_y_clamped = @min(src_y, old_height - 1);
        for (0..new_width) |x| {
            const src_x = @as(usize, @intFromFloat(@as(f32, @floatFromInt(x)) * @as(f32, @floatFromInt(old_width)) / @as(f32, @floatFromInt(new_width))));
            const src_x_clamped = @min(src_x, old_width - 1);
            const src_idx = src_y_clamped * old_width + src_x_clamped;
            const dst_idx = y * new_width + x;
            new_pixels[dst_idx] = pixels[src_idx];
        }
    }

    // Update image with new dimensions and pixels
    ctx.allocator.free(ctx.image.pixels.rgba32);
    ctx.image.width = new_width;
    ctx.image.height = new_height;
    ctx.image.pixels.rgba32 = new_pixels;

    utils.logMemoryUsage(ctx, "Resize end");
}

pub fn cropImage(ctx: *Context, args: anytype) !void {
    const x = args[0];
    const y = args[1];
    const width = args[2];
    const height = args[3];

    // Input validation
    if (x < 0 or y < 0 or width <= 0 or height <= 0) {
        return error.InvalidCropParameters;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Cropping image at ({},{}) with size {}x{}", .{ x, y, width, height });

    const pixels = ctx.image.pixels.rgba32;
    const old_width = ctx.image.width;
    const old_height = ctx.image.height;

    // Clamp crop region to image bounds
    const crop_x = @min(@as(usize, @intCast(x)), old_width);
    const crop_y = @min(@as(usize, @intCast(y)), old_height);
    const crop_width = @min(@as(usize, @intCast(width)), old_width - crop_x);
    const crop_height = @min(@as(usize, @intCast(height)), old_height - crop_y);

    if (crop_width == 0 or crop_height == 0) {
        return error.EmptyCropRegion;
    }

    // Create new cropped image
    const size = std.math.mul(usize, crop_width, crop_height) catch return error.ImageTooLarge;
    const new_pixels = try ctx.allocator.alloc(img.color.Rgba32, size);

    // Copy pixels from crop region
    for (0..crop_height) |cy| {
        const src_y = crop_y + cy;
        const src_start = src_y * old_width + crop_x;
        const dst_start = cy * crop_width;

        @memcpy(new_pixels[dst_start .. dst_start + crop_width], pixels[src_start .. src_start + crop_width]);
    }

    // Update image with cropped dimensions and pixels
    ctx.allocator.free(ctx.image.pixels.rgba32);
    ctx.image.width = crop_width;
    ctx.image.height = crop_height;
    ctx.image.pixels.rgba32 = new_pixels;
}

pub fn flipHorizontalImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Flipping image horizontally", .{});

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try utils.createTempBufferReuse(ctx.allocator, pixels, ctx.temp_buffer);
    defer {
        ctx.temp_buffer = temp_pixels;
    }

    // Flip horizontally by swapping pixels
    for (0..height) |y| {
        for (0..width / 2) |x| {
            const left_idx = y * width + x;
            const right_idx = y * width + (width - 1 - x);

            pixels[left_idx] = temp_pixels[right_idx];
            pixels[right_idx] = temp_pixels[left_idx];
        }
    }
}

pub fn flipVerticalImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Flipping image vertically", .{});

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try utils.createTempBufferReuse(ctx.allocator, pixels, ctx.temp_buffer);
    defer {
        ctx.temp_buffer = temp_pixels;
    }

    // Flip vertically by swapping rows
    for (0..height / 2) |y| {
        const top_row = y * width;
        const bottom_row = (height - 1 - y) * width;

        for (0..width) |x| {
            pixels[top_row + x] = temp_pixels[bottom_row + x];
            pixels[bottom_row + x] = temp_pixels[top_row + x];
        }
    }
}

pub fn rotate90ClockwiseImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Rotating image 90 degrees clockwise", .{});
    utils.logMemoryUsage(ctx, "Rotate 90 start");

    const pixels = ctx.image.pixels.rgba32;
    const old_width = ctx.image.width;
    const old_height = ctx.image.height;

    // Create new image with swapped dimensions
    const new_width = old_height;
    const new_height = old_width;
    const new_pixels = try ctx.allocator.alloc(img.color.Rgba32, new_width * new_height);

    // Initialize all pixels to transparent
    @memset(new_pixels, img.color.Rgba32{ .r = 0, .g = 0, .b = 0, .a = 0 });

    // Rotate 90 degrees clockwise
    for (0..old_height) |y| {
        for (0..old_width) |x| {
            const src_idx = y * old_width + x;
            const dst_x = old_height - 1 - y;
            const dst_y = x;
            const dst_idx = dst_y * new_width + dst_x;

            new_pixels[dst_idx] = pixels[src_idx];
        }
    }

    // Update image with rotated dimensions and pixels
    ctx.allocator.free(ctx.image.pixels.rgba32);
    ctx.image.width = new_width;
    ctx.image.height = new_height;
    ctx.image.pixels.rgba32 = new_pixels;

    utils.logMemoryUsage(ctx, "Rotate 90 end");
}

pub fn rotate180Image(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Rotating image 180 degrees", .{});

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try utils.createTempBufferReuse(ctx.allocator, pixels, ctx.temp_buffer);
    defer {
        ctx.temp_buffer = temp_pixels;
    }

    // Rotate 180 degrees by reversing all pixels
    for (0..height) |y| {
        for (0..width) |x| {
            const src_idx = y * width + x;
            const dst_idx = (height - 1 - y) * width + (width - 1 - x);

            pixels[dst_idx] = temp_pixels[src_idx];
        }
    }
}

pub fn rotate270ClockwiseImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Rotating image 270 degrees clockwise (90 degrees counter-clockwise)", .{});

    const pixels = ctx.image.pixels.rgba32;
    const old_width = ctx.image.width;
    const old_height = ctx.image.height;

    // Create new image with swapped dimensions
    const new_width = old_height;
    const new_height = old_width;
    const new_pixels = try ctx.allocator.alloc(img.color.Rgba32, new_width * new_height);

    // Initialize all pixels to transparent
    @memset(new_pixels, img.color.Rgba32{ .r = 0, .g = 0, .b = 0, .a = 0 });

    // Rotate 270 degrees clockwise (90 degrees counter-clockwise)
    for (0..old_height) |y| {
        for (0..old_width) |x| {
            const src_idx = y * old_width + x;
            const dst_x = y;
            const dst_y = old_width - 1 - x;
            const dst_idx = dst_y * new_width + dst_x;

            new_pixels[dst_idx] = pixels[src_idx];
        }
    }

    // Update image with rotated dimensions and pixels
    ctx.allocator.free(ctx.image.pixels.rgba32);
    ctx.image.width = new_width;
    ctx.image.height = new_height;
    ctx.image.pixels.rgba32 = new_pixels;
}

pub fn rotateArbitraryImage(ctx: *Context, degrees: f64) !void {
    try utils.convertToRgba32(ctx);

    const radians = degrees * std.math.pi / 180.0;
    const cos_theta = @cos(radians);
    const sin_theta = @sin(radians);

    const old_width = ctx.image.width;
    const old_height = ctx.image.height;

    // Calculate the bounding box of the rotated image
    const corners = [_][2]f64{
        .{ 0, 0 },
        .{ @as(f64, @floatFromInt(old_width)), 0 },
        .{ 0, @as(f64, @floatFromInt(old_height)) },
        .{ @as(f64, @floatFromInt(old_width)), @as(f64, @floatFromInt(old_height)) },
    };

    var min_x: f64 = 0;
    var max_x: f64 = 0;
    var min_y: f64 = 0;
    var max_y: f64 = 0;

    for (corners) |corner| {
        const x = corner[0];
        const y = corner[1];

        // Apply rotation matrix
        const rotated_x = x * cos_theta - y * sin_theta;
        const rotated_y = x * sin_theta + y * cos_theta;

        min_x = @min(min_x, rotated_x);
        max_x = @max(max_x, rotated_x);
        min_y = @min(min_y, rotated_y);
        max_y = @max(max_y, rotated_y);
    }

    const new_width = @as(usize, @intFromFloat(@ceil(max_x - min_x)));
    const new_height = @as(usize, @intFromFloat(@ceil(max_y - min_y)));

    if (new_width == 0 or new_height == 0) {
        return error.InvalidDimensions;
    }

    utils.logVerbose(ctx, "Rotating image by {} degrees, new size: {}x{}", .{ degrees, new_width, new_height });
    utils.logMemoryUsage(ctx, "Arbitrary rotate start");

    const pixels = ctx.image.pixels.rgba32;
    const new_pixels = try ctx.allocator.alloc(img.color.Rgba32, new_width * new_height);

    // Initialize all pixels to transparent
    @memset(new_pixels, img.color.Rgba32{ .r = 0, .g = 0, .b = 0, .a = 0 });

    // Calculate offset to center the rotated image
    const offset_x = -min_x;
    const offset_y = -min_y;

    // For each pixel in the new image
    for (0..new_height) |new_y| {
        for (0..new_width) |new_x| {
            // Convert to coordinate system centered at origin
            const centered_x = @as(f64, @floatFromInt(new_x)) - offset_x;
            const centered_y = @as(f64, @floatFromInt(new_y)) - offset_y;

            // Apply inverse rotation
            const orig_x = centered_x * cos_theta + centered_y * sin_theta;
            const orig_y = -centered_x * sin_theta + centered_y * cos_theta;

            // Only write pixels that map to valid source coordinates
            if (orig_x >= 0 and orig_x < @as(f64, @floatFromInt(old_width)) and
                orig_y >= 0 and orig_y < @as(f64, @floatFromInt(old_height)))
            {
                const x0 = @as(i32, @intFromFloat(@floor(orig_x)));
                const y0 = @as(i32, @intFromFloat(@floor(orig_y)));
                const x1 = x0 + 1;
                const y1 = y0 + 1;

                const wx = orig_x - @as(f64, @floatFromInt(x0));
                const wy = orig_y - @as(f64, @floatFromInt(y0));

                // Get the four neighboring pixels
                const p00 = utils.getPixelClamped(pixels, old_width, old_height, x0, y0);
                const p01 = utils.getPixelClamped(pixels, old_width, old_height, x0, y1);
                const p10 = utils.getPixelClamped(pixels, old_width, old_height, x1, y0);
                const p11 = utils.getPixelClamped(pixels, old_width, old_height, x1, y1);

                // Bilinear interpolation for each channel
                const r = @as(u8, @intFromFloat((1 - wx) * (1 - wy) * @as(f64, @floatFromInt(p00.r)) +
                    wx * (1 - wy) * @as(f64, @floatFromInt(p10.r)) +
                    (1 - wx) * wy * @as(f64, @floatFromInt(p01.r)) +
                    wx * wy * @as(f64, @floatFromInt(p11.r))));

                const g = @as(u8, @intFromFloat((1 - wx) * (1 - wy) * @as(f64, @floatFromInt(p00.g)) +
                    wx * (1 - wy) * @as(f64, @floatFromInt(p10.g)) +
                    (1 - wx) * wy * @as(f64, @floatFromInt(p01.g)) +
                    wx * wy * @as(f64, @floatFromInt(p11.g))));

                const b = @as(u8, @intFromFloat((1 - wx) * (1 - wy) * @as(f64, @floatFromInt(p00.b)) +
                    wx * (1 - wy) * @as(f64, @floatFromInt(p10.b)) +
                    (1 - wx) * wy * @as(f64, @floatFromInt(p01.b)) +
                    wx * wy * @as(f64, @floatFromInt(p11.b))));

                const a = @as(u8, @intFromFloat((1 - wx) * (1 - wy) * @as(f64, @floatFromInt(p00.a)) +
                    wx * (1 - wy) * @as(f64, @floatFromInt(p10.a)) +
                    (1 - wx) * wy * @as(f64, @floatFromInt(p01.a)) +
                    wx * wy * @as(f64, @floatFromInt(p11.a))));

                new_pixels[new_y * new_width + new_x] = img.color.Rgba32{ .r = r, .g = g, .b = b, .a = a };
            }
        }
    }

    // Update image with new dimensions and pixels
    ctx.allocator.free(ctx.image.pixels.rgba32);
    ctx.image.width = new_width;
    ctx.image.height = new_height;
    ctx.image.pixels.rgba32 = new_pixels;

    utils.logMemoryUsage(ctx, "Arbitrary rotate end");
}

pub fn rotateImage(ctx: *Context, args: anytype) !void {
    const degrees = args[0];

    // Input validation
    if (degrees < -360 or degrees > 360) {
        return error.InvalidParameters;
    }

    const normalized_degrees = @mod(degrees, 360.0);

    if (normalized_degrees == 0) {
        // No rotation needed
        utils.logVerbose(ctx, "No rotation needed (0 degrees)", .{});
        return;
    }

    // Check if it's a multiple of 90 degrees for optimized rotation
    const is_multiple_of_90 = @abs(@rem(normalized_degrees, 90.0)) < 0.001;

    if (is_multiple_of_90) {
        if (normalized_degrees == 90) {
            try rotate90ClockwiseImage(ctx, .{});
        } else if (normalized_degrees == 180) {
            try rotate180Image(ctx, .{});
        } else if (normalized_degrees == 270) {
            try rotate270ClockwiseImage(ctx, .{});
        }
    } else {
        // Use arbitrary angle rotation
        try rotateArbitraryImage(ctx, normalized_degrees);
    }
}

pub fn mirrorHorizontalImage(ctx: *Context, args: anytype) !void {
    _ = args;
    // Mirror horizontal is the same as flip horizontal
    try flipHorizontalImage(ctx, .{});
}

pub fn mirrorVerticalImage(ctx: *Context, args: anytype) !void {
    _ = args;
    // Mirror vertical is the same as flip vertical
    try flipVerticalImage(ctx, .{});
}
