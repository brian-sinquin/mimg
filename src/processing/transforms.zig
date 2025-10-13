const img = @import("zigimg");
const std = @import("std");
const Context = @import("../core/types.zig").Context;
const utils = @import("../core/utils.zig");
const math = std.math;
const simd = @import("../utils/simd_utils.zig");

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

pub fn roundCornersImage(ctx: *Context, args: anytype) !void {
    const radius = args[0];

    // Input validation
    if (radius < 0) {
        std.log.err("Corner radius must be non-negative, got {d}", .{radius});
        return error.InvalidRadius;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Rounding corners with radius {d}", .{radius});

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // If radius is 0, do nothing
    if (radius == 0) return;

    const radius_f = @as(f32, @floatFromInt(radius));
    const width_f = @as(f32, @floatFromInt(width));
    const height_f = @as(f32, @floatFromInt(height));

    const pixel_count = pixels.len;
    var i: usize = 0;

    // Process in chunks of 4 pixels using SIMD
    while (i + 4 <= pixel_count) : (i += 4) {
        // Calculate coordinates for 4 pixels
        const y0 = i / width;
        const x0 = i % width;
        const y1 = (i + 1) / width;
        const x1 = (i + 1) % width;
        const y2 = (i + 2) / width;
        const x2 = (i + 2) % width;
        const y3 = (i + 3) / width;
        const x3 = (i + 3) % width;

        // Load pixel values
        const r_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].r)),
            @as(f32, @floatFromInt(pixels[i + 1].r)),
            @as(f32, @floatFromInt(pixels[i + 2].r)),
            @as(f32, @floatFromInt(pixels[i + 3].r)),
        };
        const g_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].g)),
            @as(f32, @floatFromInt(pixels[i + 1].g)),
            @as(f32, @floatFromInt(pixels[i + 2].g)),
            @as(f32, @floatFromInt(pixels[i + 3].g)),
        };
        const b_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].b)),
            @as(f32, @floatFromInt(pixels[i + 1].b)),
            @as(f32, @floatFromInt(pixels[i + 2].b)),
            @as(f32, @floatFromInt(pixels[i + 3].b)),
        };
        const a_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].a)),
            @as(f32, @floatFromInt(pixels[i + 1].a)),
            @as(f32, @floatFromInt(pixels[i + 2].a)),
            @as(f32, @floatFromInt(pixels[i + 3].a)),
        };

        // Calculate coordinates as f32 vectors
        const x_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(x0)),
            @as(f32, @floatFromInt(x1)),
            @as(f32, @floatFromInt(x2)),
            @as(f32, @floatFromInt(x3)),
        };
        const y_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(y0)),
            @as(f32, @floatFromInt(y1)),
            @as(f32, @floatFromInt(y2)),
            @as(f32, @floatFromInt(y3)),
        };

        // Apply round corners effect using SIMD
        const result = simd.applyRoundCornersSIMD4(r_vec, g_vec, b_vec, a_vec, x_vec, y_vec, width_f, height_f, radius_f);
        const new_a_vec = result[3];

        // Convert back to u8 and store (RGB unchanged, only alpha modified)
        pixels[i].a = @as(u8, @intFromFloat(math.clamp(new_a_vec[0], 0.0, 255.0)));
        pixels[i + 1].a = @as(u8, @intFromFloat(math.clamp(new_a_vec[1], 0.0, 255.0)));
        pixels[i + 2].a = @as(u8, @intFromFloat(math.clamp(new_a_vec[2], 0.0, 255.0)));
        pixels[i + 3].a = @as(u8, @intFromFloat(math.clamp(new_a_vec[3], 0.0, 255.0)));
    }

    // Handle remaining pixels with scalar operations
    while (i < pixel_count) : (i += 1) {
        const y = i / width;
        const x = i % width;

        const y_f = @as(f32, @floatFromInt(y));
        const x_f = @as(f32, @floatFromInt(x));
        const dist_from_top = y_f;
        const dist_from_bottom = @as(f32, @floatFromInt(height - 1)) - y_f;
        const dist_from_left = x_f;
        const dist_from_right = @as(f32, @floatFromInt(width - 1)) - x_f;

        // Check if we're in a corner region
        var alpha_mult: f32 = 1.0;

        // Top-left corner
        if (dist_from_left <= radius_f and dist_from_top <= radius_f) {
            const dx = radius_f - dist_from_left;
            const dy = radius_f - dist_from_top;
            const dist_from_corner = math.sqrt(dx * dx + dy * dy);
            if (dist_from_corner > radius_f) {
                alpha_mult = 0.0;
            } else if (dist_from_corner > radius_f - 1.0) {
                // Anti-aliasing: smooth transition
                alpha_mult = (radius_f - dist_from_corner);
            }
        }
        // Top-right corner
        else if (dist_from_right <= radius_f and dist_from_top <= radius_f) {
            const dx = radius_f - dist_from_right;
            const dy = radius_f - dist_from_top;
            const dist_from_corner = math.sqrt(dx * dx + dy * dy);
            if (dist_from_corner > radius_f) {
                alpha_mult = 0.0;
            } else if (dist_from_corner > radius_f - 1.0) {
                alpha_mult = (radius_f - dist_from_corner);
            }
        }
        // Bottom-left corner
        else if (dist_from_left <= radius_f and dist_from_bottom <= radius_f) {
            const dx = radius_f - dist_from_left;
            const dy = radius_f - dist_from_bottom;
            const dist_from_corner = math.sqrt(dx * dx + dy * dy);
            if (dist_from_corner > radius_f) {
                alpha_mult = 0.0;
            } else if (dist_from_corner > radius_f - 1.0) {
                alpha_mult = (radius_f - dist_from_corner);
            }
        }
        // Bottom-right corner
        else if (dist_from_right <= radius_f and dist_from_bottom <= radius_f) {
            const dx = radius_f - dist_from_right;
            const dy = radius_f - dist_from_bottom;
            const dist_from_corner = math.sqrt(dx * dx + dy * dy);
            if (dist_from_corner > radius_f) {
                alpha_mult = 0.0;
            } else if (dist_from_corner > radius_f - 1.0) {
                alpha_mult = (radius_f - dist_from_corner);
            }
        }

        // Apply alpha multiplier
        const current_alpha = @as(f32, @floatFromInt(pixels[i].a)) / 255.0;
        const new_alpha = current_alpha * alpha_mult;
        pixels[i].a = @as(u8, @intFromFloat(math.clamp(new_alpha * 255.0, 0.0, 255.0)));
    }
}
