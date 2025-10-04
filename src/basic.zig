const img = @import("zigimg");
const std = @import("std");
const Context = @import("types.zig").Context;
const math = std.math;

pub fn invertColors(ctx: *Context, args: anytype) !void {
    _ = args;
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying invert modifier", .{});
    }

    for (ctx.image.pixels.rgba32) |*pixel| {
        const r = pixel.r;
        const g = pixel.g;
        const b = pixel.b;

        pixel.r = 255 - r;
        pixel.g = 255 - g;
        pixel.b = 255 - b;
    }
}

pub fn resizeImage(ctx: *Context, args: anytype) !void {
    // Simple nearest-neighbor resize implementation
    const new_width = args[0];
    const new_height = args[1];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info(
            "Applying resize modifier to {}x{}",
            .{ new_width, new_height },
        );
    }

    const old_width = ctx.image.width;
    const old_height = ctx.image.height;
    const old_pixels = ctx.image.pixels.rgba32;

    const resized = try img.Image.create(
        ctx.allocator,
        new_width,
        new_height,
        .rgba32,
    );

    for (resized.pixels.rgba32, 0..) |*out_pixel, i| {
        const x = @as(f32, @floatFromInt(@mod(i, @as(usize, new_width)))) * @as(f32, @floatFromInt(old_width)) / @as(f32, @floatFromInt(new_width));
        const y = @as(f32, @floatFromInt(i / @as(usize, new_width))) * @as(f32, @floatFromInt(old_height)) / @as(f32, @floatFromInt(new_height));

        const src_x = @as(usize, @intFromFloat(x));
        const src_y = @as(usize, @intFromFloat(y));

        const src_index = src_y * old_width + src_x;
        out_pixel.* = old_pixels[src_index];
    }
    ctx.setImage(resized);
}

pub fn cropImage(ctx: *Context, args: anytype) !void {
    const x = args[0];
    const y = args[1];
    const w = args[2];
    const h = args[3];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info(
            "Applying crop modifier at ({}, {}) size {}x{}",
            .{ x, y, w, h },
        );
    }

    const old_width = ctx.image.width;
    const old_pixels = ctx.image.pixels.rgba32;

    const resized = try img.Image.create(
        ctx.allocator,
        w,
        h,
        .rgba32,
    );

    for (resized.pixels.rgba32, 0..) |*out_pixel, i| {
        const newX = x + i % w;
        const newY = y + i / w;

        const src_index = (newY * old_width) + newX;
        out_pixel.* = old_pixels[src_index];
    }
    ctx.setImage(resized);
}

pub fn rotateImage(ctx: *Context, args: anytype) !void {
    const angle_degrees = args[0];
    const normalized = @mod(angle_degrees, 360.0);

    if (math.approxEqAbs(f64, normalized, 0.0, 1e-6)) {
        if (ctx.verbose) {
            std.log.info("Rotate modifier received 0 degrees, skipping", .{});
        }
        return;
    }

    try ctx.image.convert(ctx.allocator, .rgba32);

    const old_width = ctx.image.width;
    const old_height = ctx.image.height;
    const src_pixels = ctx.image.pixels.rgba32;

    const angle_rad = angle_degrees * math.pi / 180.0;
    const cos_theta = math.cos(angle_rad);
    const sin_theta = math.sin(angle_rad);

    const old_wf = @as(f64, @floatFromInt(old_width));
    const old_hf = @as(f64, @floatFromInt(old_height));

    const abs_cos = @abs(cos_theta);
    const abs_sin = @abs(sin_theta);

    const new_wf = abs_cos * old_wf + abs_sin * old_hf;
    const new_hf = abs_sin * old_wf + abs_cos * old_hf;

    const new_width = @max(1, @as(usize, @intFromFloat(math.ceil(new_wf))));
    const new_height = @max(1, @as(usize, @intFromFloat(math.ceil(new_hf))));

    const new_width_u32 = @as(u32, @intCast(new_width));
    const new_height_u32 = @as(u32, @intCast(new_height));

    const rotated = try img.Image.create(
        ctx.allocator,
        new_width_u32,
        new_height_u32,
        .rgba32,
    );

    const dst_pixels = rotated.pixels.rgba32;

    const src_cx = (old_wf - 1.0) / 2.0;
    const src_cy = (old_hf - 1.0) / 2.0;
    const dst_cx = (@as(f64, @floatFromInt(new_width)) - 1.0) / 2.0;
    const dst_cy = (@as(f64, @floatFromInt(new_height)) - 1.0) / 2.0;

    for (dst_pixels, 0..) |*dst_pixel, idx| {
        dst_pixel.* = .{ .r = 0, .g = 0, .b = 0, .a = 0 };

        const dest_x = @as(f64, @floatFromInt(idx % new_width));
        const dest_y = @as(f64, @floatFromInt(idx / new_width));

        const dx = dest_x - dst_cx;
        const dy = dest_y - dst_cy;

        const src_x = cos_theta * dx + sin_theta * dy + src_cx;
        const src_y = -sin_theta * dx + cos_theta * dy + src_cy;

        const src_x_round = math.round(src_x);
        const src_y_round = math.round(src_y);

        if (src_x_round < 0 or src_y_round < 0) continue;

        const sx: usize = @intFromFloat(src_x_round);
        const sy: usize = @intFromFloat(src_y_round);

        if (sx >= old_width or sy >= old_height) continue;

        const src_index = sy * old_width + sx;
        dst_pixel.* = src_pixels[src_index];
    }

    if (ctx.verbose) {
        std.log.info(
            "Applying rotate modifier by {d:.2} degrees -> new size {}x{}",
            .{ angle_degrees, new_width, new_height },
        );
    }

    ctx.setImage(rotated);
}
