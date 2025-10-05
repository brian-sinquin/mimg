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

pub fn flipImage(ctx: *Context, args: anytype) !void {
    const direction = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying flip {s} modifier", .{direction});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    if (std.mem.eql(u8, direction, "horizontal")) {
        for (0..height) |y| {
            for (0..width / 2) |x| {
                const left = y * width + x;
                const right = y * width + (width - 1 - x);
                std.mem.swap(img.color.Rgba32, &pixels[left], &pixels[right]);
            }
        }
    } else if (std.mem.eql(u8, direction, "vertical")) {
        for (0..height / 2) |y| {
            for (0..width) |x| {
                const top = y * width + x;
                const bottom = (height - 1 - y) * width + x;
                std.mem.swap(img.color.Rgba32, &pixels[top], &pixels[bottom]);
            }
        }
    } else {
        return error.InvalidDirection;
    }
}

pub fn grayscaleImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying grayscale modifier", .{});
    }

    for (ctx.image.pixels.rgba32) |*pixel| {
        const gray = @as(u8, @intFromFloat(0.299 * @as(f32, @floatFromInt(pixel.r)) + 0.587 * @as(f32, @floatFromInt(pixel.g)) + 0.114 * @as(f32, @floatFromInt(pixel.b))));
        pixel.r = gray;
        pixel.g = gray;
        pixel.b = gray;
    }
}

pub fn adjustBrightness(ctx: *Context, args: anytype) !void {
    const delta = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying brightness adjustment by {}", .{delta});
    }

    for (ctx.image.pixels.rgba32) |*pixel| {
        pixel.r = @as(u8, @intCast(std.math.clamp(@as(i16, @intCast(pixel.r)) + delta, 0, 255)));
        pixel.g = @as(u8, @intCast(std.math.clamp(@as(i16, @intCast(pixel.g)) + delta, 0, 255)));
        pixel.b = @as(u8, @intCast(std.math.clamp(@as(i16, @intCast(pixel.b)) + delta, 0, 255)));
    }
}

pub fn blurImage(ctx: *Context, args: anytype) !void {
    const kernel_size = args[0];
    if (kernel_size % 2 == 0) return error.EvenKernelSize;
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying blur with kernel size {}", .{kernel_size});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.allocator.alloc(img.color.Rgba32, width * height);
    defer ctx.allocator.free(temp_pixels);
    @memcpy(temp_pixels, pixels);

    const half_kernel = kernel_size / 2;

    for (0..height) |y| {
        for (0..width) |x| {
            var r_sum: u32 = 0;
            var g_sum: u32 = 0;
            var b_sum: u32 = 0;
            var a_sum: u32 = 0;
            var count: u32 = 0;

            var ky: i32 = -@as(i32, @intCast(half_kernel));
            while (ky <= @as(i32, @intCast(half_kernel))) : (ky += 1) {
                var kx: i32 = -@as(i32, @intCast(half_kernel));
                while (kx <= @as(i32, @intCast(half_kernel))) : (kx += 1) {
                    const nx = @as(i32, @intCast(x)) + kx;
                    const ny = @as(i32, @intCast(y)) + ky;
                    if (nx >= 0 and nx < @as(i32, @intCast(width)) and ny >= 0 and ny < @as(i32, @intCast(height))) {
                        const idx = @as(usize, @intCast(ny)) * width + @as(usize, @intCast(nx));
                        const pixel = temp_pixels[idx];
                        r_sum += pixel.r;
                        g_sum += pixel.g;
                        b_sum += pixel.b;
                        a_sum += pixel.a;
                        count += 1;
                    }
                }
            }

            const idx = y * width + x;
            pixels[idx].r = @as(u8, @intCast(r_sum / count));
            pixels[idx].g = @as(u8, @intCast(g_sum / count));
            pixels[idx].b = @as(u8, @intCast(b_sum / count));
            pixels[idx].a = @as(u8, @intCast(a_sum / count));
        }
    }
}

pub fn adjustSaturation(ctx: *Context, args: anytype) !void {
    const factor = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying saturation adjustment by {}", .{factor});
    }

    for (ctx.image.pixels.rgba32) |*pixel| {
        var h: f32 = 0;
        var s: f32 = 0;
        var l: f32 = 0;
        rgbToHsl(pixel.r, pixel.g, pixel.b, &h, &s, &l);
        s = std.math.clamp(s * factor, 0.0, 1.0);
        hslToRgb(h, s, l, &pixel.r, &pixel.g, &pixel.b);
    }
}

pub fn adjustContrast(ctx: *Context, args: anytype) !void {
    const factor = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying contrast adjustment by {}", .{factor});
    }

    for (ctx.image.pixels.rgba32) |*pixel| {
        pixel.r = @as(u8, @intFromFloat(std.math.clamp((@as(f32, @floatFromInt(pixel.r)) - 128.0) * factor + 128.0, 0.0, 255.0)));
        pixel.g = @as(u8, @intFromFloat(std.math.clamp((@as(f32, @floatFromInt(pixel.g)) - 128.0) * factor + 128.0, 0.0, 255.0)));
        pixel.b = @as(u8, @intFromFloat(std.math.clamp((@as(f32, @floatFromInt(pixel.b)) - 128.0) * factor + 128.0, 0.0, 255.0)));
    }
}

pub fn adjustGamma(ctx: *Context, args: anytype) !void {
    const gamma = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying gamma correction with gamma {}", .{gamma});
    }

    const inv_gamma = 1.0 / gamma;

    for (ctx.image.pixels.rgba32) |*pixel| {
        pixel.r = @as(u8, @intFromFloat(255.0 * std.math.pow(f32, @as(f32, @floatFromInt(pixel.r)) / 255.0, inv_gamma)));
        pixel.g = @as(u8, @intFromFloat(255.0 * std.math.pow(f32, @as(f32, @floatFromInt(pixel.g)) / 255.0, inv_gamma)));
        pixel.b = @as(u8, @intFromFloat(255.0 * std.math.pow(f32, @as(f32, @floatFromInt(pixel.b)) / 255.0, inv_gamma)));
    }
}

pub fn applySepia(ctx: *Context, args: anytype) !void {
    _ = args;
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying sepia tone effect", .{});
    }

    for (ctx.image.pixels.rgba32) |*pixel| {
        const r = @as(f32, @floatFromInt(pixel.r));
        const g = @as(f32, @floatFromInt(pixel.g));
        const b = @as(f32, @floatFromInt(pixel.b));

        pixel.r = @as(u8, @intFromFloat(std.math.clamp(0.393 * r + 0.769 * g + 0.189 * b, 0.0, 255.0)));
        pixel.g = @as(u8, @intFromFloat(std.math.clamp(0.349 * r + 0.686 * g + 0.168 * b, 0.0, 255.0)));
        pixel.b = @as(u8, @intFromFloat(std.math.clamp(0.272 * r + 0.534 * g + 0.131 * b, 0.0, 255.0)));
    }
}

pub fn sharpenImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying sharpen effect", .{});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.allocator.alloc(img.color.Rgba32, width * height);
    defer ctx.allocator.free(temp_pixels);
    @memcpy(temp_pixels, pixels);

    // Simple sharpening using 3x3 kernel: center * 5 - orthogonal neighbors * 1
    // Skip 1-pixel border since kernel needs all neighbors
    for (1..height - 1) |y| {
        for (1..width - 1) |x| {
            var r_sum: i32 = 0;
            var g_sum: i32 = 0;
            var b_sum: i32 = 0;

            // Apply sharpen kernel: [0, -1, 0; -1, 5, -1; 0, -1, 0]
            // Center pixel
            const center_idx = y * width + x;
            r_sum += @as(i32, @intCast(temp_pixels[center_idx].r)) * 5;
            g_sum += @as(i32, @intCast(temp_pixels[center_idx].g)) * 5;
            b_sum += @as(i32, @intCast(temp_pixels[center_idx].b)) * 5;

            // Orthogonal neighbors (up, down, left, right)
            // Up
            const up_idx = (y - 1) * width + x;
            r_sum -= @as(i32, @intCast(temp_pixels[up_idx].r));
            g_sum -= @as(i32, @intCast(temp_pixels[up_idx].g));
            b_sum -= @as(i32, @intCast(temp_pixels[up_idx].b));

            // Down
            const down_idx = (y + 1) * width + x;
            r_sum -= @as(i32, @intCast(temp_pixels[down_idx].r));
            g_sum -= @as(i32, @intCast(temp_pixels[down_idx].g));
            b_sum -= @as(i32, @intCast(temp_pixels[down_idx].b));

            // Left
            const left_idx = y * width + (x - 1);
            r_sum -= @as(i32, @intCast(temp_pixels[left_idx].r));
            g_sum -= @as(i32, @intCast(temp_pixels[left_idx].g));
            b_sum -= @as(i32, @intCast(temp_pixels[left_idx].b));

            // Right
            const right_idx = y * width + (x + 1);
            r_sum -= @as(i32, @intCast(temp_pixels[right_idx].r));
            g_sum -= @as(i32, @intCast(temp_pixels[right_idx].g));
            b_sum -= @as(i32, @intCast(temp_pixels[right_idx].b));

            const idx = y * width + x;
            pixels[idx].r = @as(u8, @intCast(std.math.clamp(r_sum, 0, 255)));
            pixels[idx].g = @as(u8, @intCast(std.math.clamp(g_sum, 0, 255)));
            pixels[idx].b = @as(u8, @intCast(std.math.clamp(b_sum, 0, 255)));
        }
    }
}

fn rgbToHsl(r: u8, g: u8, b: u8, h: *f32, s: *f32, l: *f32) void {
    const rf = @as(f32, @floatFromInt(r)) / 255.0;
    const gf = @as(f32, @floatFromInt(g)) / 255.0;
    const bf = @as(f32, @floatFromInt(b)) / 255.0;

    const max = @max(rf, @max(gf, bf));
    const min = @min(rf, @min(gf, bf));
    const delta = max - min;

    l.* = (max + min) / 2.0;

    if (delta == 0) {
        h.* = 0;
        s.* = 0;
    } else {
        s.* = if (l.* < 0.5) delta / (max + min) else delta / (2.0 - max - min);

        if (max == rf) {
            h.* = (gf - bf) / delta;
        } else if (max == gf) {
            h.* = 2.0 + (bf - rf) / delta;
        } else {
            h.* = 4.0 + (rf - gf) / delta;
        }
        h.* /= 6.0;
        if (h.* < 0) h.* += 1.0;
    }
}

fn hslToRgb(h: f32, s: f32, l: f32, r: *u8, g: *u8, b: *u8) void {
    if (s == 0) {
        const val = @as(u8, @intFromFloat(l * 255.0));
        r.* = val;
        g.* = val;
        b.* = val;
    } else {
        const q = if (l < 0.5) l * (1.0 + s) else l + s - l * s;
        const p = 2.0 * l - q;

        const hk = h * 6.0;
        const tr = hk + 2.0 / 3.0;
        const tg = hk;
        const tb = hk - 2.0 / 3.0;

        r.* = @as(u8, @intFromFloat(hueToRgb(p, q, tr) * 255.0));
        g.* = @as(u8, @intFromFloat(hueToRgb(p, q, tg) * 255.0));
        b.* = @as(u8, @intFromFloat(hueToRgb(p, q, tb) * 255.0));
    }
}

fn hueToRgb(p: f32, q: f32, t: f32) f32 {
    var tt = t;
    if (tt < 0) tt += 1.0;
    if (tt > 1) tt -= 1.0;
    if (tt < 1.0 / 6.0) return p + (q - p) * 6.0 * tt;
    if (tt < 1.0 / 2.0) return q;
    if (tt < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - tt) * 6.0;
    return p;
}

pub fn gaussianBlurImage(ctx: *Context, args: anytype) !void {
    const sigma = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying Gaussian blur with sigma {d:.2}", .{sigma});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.allocator.alloc(img.color.Rgba32, width * height);
    defer ctx.allocator.free(temp_pixels);
    @memcpy(temp_pixels, pixels);

    // Calculate kernel size (should be odd and cover ~3*sigma)
    const kernel_radius = @as(usize, @intFromFloat(@ceil(sigma * 3.0)));
    const kernel_size = kernel_radius * 2 + 1;

    // Generate Gaussian kernel
    const kernel = try ctx.allocator.alloc(f32, kernel_size * kernel_size);
    defer ctx.allocator.free(kernel);

    var sum: f32 = 0.0;
    const two_sigma_sq = 2.0 * sigma * sigma;

    for (0..kernel_size) |i| {
        for (0..kernel_size) |j| {
            const x = @as(f32, @floatFromInt(i)) - @as(f32, @floatFromInt(kernel_radius));
            const y = @as(f32, @floatFromInt(j)) - @as(f32, @floatFromInt(kernel_radius));
            const weight = std.math.exp(-(x * x + y * y) / two_sigma_sq);
            kernel[i * kernel_size + j] = weight;
            sum += weight;
        }
    }

    // Normalize kernel
    for (kernel) |*w| {
        w.* /= sum;
    }

    // Apply convolution
    for (0..height) |y| {
        for (0..width) |x| {
            var r_sum: f32 = 0.0;
            var g_sum: f32 = 0.0;
            var b_sum: f32 = 0.0;
            var a_sum: f32 = 0.0;

            for (0..kernel_size) |ky| {
                for (0..kernel_size) |kx| {
                    const px = @as(i32, @intCast(x)) + @as(i32, @intCast(kx)) - @as(i32, @intCast(kernel_radius));
                    const py = @as(i32, @intCast(y)) + @as(i32, @intCast(ky)) - @as(i32, @intCast(kernel_radius));

                    if (px >= 0 and px < width and py >= 0 and py < height) {
                        const idx = @as(usize, @intCast(py)) * width + @as(usize, @intCast(px));
                        const weight = kernel[ky * kernel_size + kx];
                        r_sum += @as(f32, @floatFromInt(temp_pixels[idx].r)) * weight;
                        g_sum += @as(f32, @floatFromInt(temp_pixels[idx].g)) * weight;
                        b_sum += @as(f32, @floatFromInt(temp_pixels[idx].b)) * weight;
                        a_sum += @as(f32, @floatFromInt(temp_pixels[idx].a)) * weight;
                    }
                }
            }

            const idx = y * width + x;
            pixels[idx].r = @as(u8, @intFromFloat(std.math.clamp(r_sum, 0.0, 255.0)));
            pixels[idx].g = @as(u8, @intFromFloat(std.math.clamp(g_sum, 0.0, 255.0)));
            pixels[idx].b = @as(u8, @intFromFloat(std.math.clamp(b_sum, 0.0, 255.0)));
            pixels[idx].a = @as(u8, @intFromFloat(std.math.clamp(a_sum, 0.0, 255.0)));
        }
    }
}

pub fn embossImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying emboss effect", .{});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.allocator.alloc(img.color.Rgba32, width * height);
    defer ctx.allocator.free(temp_pixels);
    @memcpy(temp_pixels, pixels);

    // Emboss kernel: [-2, -1, 0; -1, 1, 1; 0, 1, 2]
    // Process all pixels, using edge replication for border pixels
    for (0..height) |y| {
        for (0..width) |x| {
            const center_idx = y * width + x;
            const center_gray = @as(i32, @intCast(temp_pixels[center_idx].r)) * 77 +
                @as(i32, @intCast(temp_pixels[center_idx].g)) * 150 +
                @as(i32, @intCast(temp_pixels[center_idx].b)) * 29;

            // Apply emboss kernel with edge replication
            var sum: i32 = 0;

            // Helper function to get pixel with edge replication
            const getPixelGray = struct {
                fn get(px: []const img.color.Rgba32, w: usize, h: usize, x_pos: i32, y_pos: i32) i32 {
                    const clamped_x = std.math.clamp(x_pos, 0, @as(i32, @intCast(w - 1)));
                    const clamped_y = std.math.clamp(y_pos, 0, @as(i32, @intCast(h - 1)));
                    const idx = @as(usize, @intCast(clamped_y)) * w + @as(usize, @intCast(clamped_x));
                    return @as(i32, @intCast(px[idx].r)) * 77 +
                        @as(i32, @intCast(px[idx].g)) * 150 +
                        @as(i32, @intCast(px[idx].b)) * 29;
                }
            }.get;

            // Top-left: -2
            sum += getPixelGray(temp_pixels, width, height, @as(i32, @intCast(x)) - 1, @as(i32, @intCast(y)) - 1) * (-2);
            // Top: -1
            sum += getPixelGray(temp_pixels, width, height, @as(i32, @intCast(x)), @as(i32, @intCast(y)) - 1) * (-1);
            // Left: -1
            sum += getPixelGray(temp_pixels, width, height, @as(i32, @intCast(x)) - 1, @as(i32, @intCast(y))) * (-1);
            // Bottom-right: 2
            sum += getPixelGray(temp_pixels, width, height, @as(i32, @intCast(x)) + 1, @as(i32, @intCast(y)) + 1) * 2;
            // Bottom: 1
            sum += getPixelGray(temp_pixels, width, height, @as(i32, @intCast(x)), @as(i32, @intCast(y)) + 1) * 1;
            // Right: 1
            sum += getPixelGray(temp_pixels, width, height, @as(i32, @intCast(x)) + 1, @as(i32, @intCast(y))) * 1;

            // Center: 1
            sum += center_gray * 1;

            // Normalize and add 128 to create the emboss effect
            const embossed = @as(i32, @intFromFloat(@as(f32, @floatFromInt(sum)) / 1024.0)) + 128;
            const clamped = std.math.clamp(embossed, 0, 255);
            const gray_value = @as(u8, @intCast(clamped));

            pixels[center_idx].r = gray_value;
            pixels[center_idx].g = gray_value;
            pixels[center_idx].b = gray_value;
        }
    }
}

pub fn vignetteImage(ctx: *Context, args: anytype) !void {
    const intensity = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying vignette effect with intensity {d:.2}", .{intensity});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    const center_x = @as(f32, @floatFromInt(width)) / 2.0;
    const center_y = @as(f32, @floatFromInt(height)) / 2.0;
    const max_distance = @sqrt(center_x * center_x + center_y * center_y);

    for (0..height) |y| {
        for (0..width) |x| {
            const dx = @as(f32, @floatFromInt(x)) - center_x;
            const dy = @as(f32, @floatFromInt(y)) - center_y;
            const distance = @sqrt(dx * dx + dy * dy);
            const normalized_distance = distance / max_distance;

            // Vignette factor: closer to center = 1.0, edges = intensity
            const vignette_factor = 1.0 - (normalized_distance * intensity);

            const idx = y * width + x;
            pixels[idx].r = @as(u8, @intFromFloat(@as(f32, @floatFromInt(pixels[idx].r)) * vignette_factor));
            pixels[idx].g = @as(u8, @intFromFloat(@as(f32, @floatFromInt(pixels[idx].g)) * vignette_factor));
            pixels[idx].b = @as(u8, @intFromFloat(@as(f32, @floatFromInt(pixels[idx].b)) * vignette_factor));
        }
    }
}

pub fn posterizeImage(ctx: *Context, args: anytype) !void {
    const levels = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying posterize effect with {d} levels", .{levels});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    const step = @as(f32, @floatFromInt(255)) / @as(f32, @floatFromInt(levels - 1));

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;

            // Posterize each channel
            const r_level = @as(u8, @intFromFloat(@round(@as(f32, @floatFromInt(pixels[idx].r)) / step) * step));
            const g_level = @as(u8, @intFromFloat(@round(@as(f32, @floatFromInt(pixels[idx].g)) / step) * step));
            const b_level = @as(u8, @intFromFloat(@round(@as(f32, @floatFromInt(pixels[idx].b)) / step) * step));

            pixels[idx].r = r_level;
            pixels[idx].g = g_level;
            pixels[idx].b = b_level;
        }
    }
}

pub fn hueShiftImage(ctx: *Context, args: anytype) !void {
    const hue_shift = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying hue shift by {d:.2} degrees", .{hue_shift});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            var h: f32 = undefined;
            var s: f32 = undefined;
            var l: f32 = undefined;

            rgbToHsl(pixels[idx].r, pixels[idx].g, pixels[idx].b, &h, &s, &l);

            // Shift hue (normalize to 0-1 range)
            h += hue_shift / 360.0;
            if (h < 0) h += 1.0;
            if (h > 1) h -= 1.0;

            hslToRgb(h, s, l, &pixels[idx].r, &pixels[idx].g, &pixels[idx].b);
        }
    }
}

pub fn medianFilterImage(ctx: *Context, args: anytype) !void {
    const kernel_size = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying median filter with kernel size {d}", .{kernel_size});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.allocator.alloc(img.color.Rgba32, width * height);
    defer ctx.allocator.free(temp_pixels);
    @memcpy(temp_pixels, pixels);

    const radius = kernel_size / 2;
    const window_size = kernel_size * kernel_size;

    // Create arrays to hold pixel values for sorting
    const r_values = try ctx.allocator.alloc(u8, window_size);
    defer ctx.allocator.free(r_values);
    const g_values = try ctx.allocator.alloc(u8, window_size);
    defer ctx.allocator.free(g_values);
    const b_values = try ctx.allocator.alloc(u8, window_size);
    defer ctx.allocator.free(b_values);

    // Skip border pixels that don't have full kernel coverage
    for (radius..height - radius) |y| {
        for (radius..width - radius) |x| {
            var count: usize = 0;

            // Collect all pixels in the kernel window
            for (0..kernel_size) |ky| {
                for (0..kernel_size) |kx| {
                    const px = x + kx - radius;
                    const py = y + ky - radius;
                    const idx = py * width + px;

                    r_values[count] = temp_pixels[idx].r;
                    g_values[count] = temp_pixels[idx].g;
                    b_values[count] = temp_pixels[idx].b;
                    count += 1;
                }
            }

            // Sort the arrays to find median
            std.sort.insertion(u8, r_values[0..window_size], {}, std.sort.asc(u8));
            std.sort.insertion(u8, g_values[0..window_size], {}, std.sort.asc(u8));
            std.sort.insertion(u8, b_values[0..window_size], {}, std.sort.asc(u8));

            const median_idx = window_size / 2;
            const idx = y * width + x;
            pixels[idx].r = r_values[median_idx];
            pixels[idx].g = g_values[median_idx];
            pixels[idx].b = b_values[median_idx];
        }
    }
}
