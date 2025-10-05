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
        const rf = @as(f32, @floatFromInt(pixel.r));
        const gf = @as(f32, @floatFromInt(pixel.g));
        const bf = @as(f32, @floatFromInt(pixel.b));

        // Calculate luminance using standard coefficients
        const gray = 0.299 * rf + 0.587 * gf + 0.114 * bf;

        // Adjust saturation by interpolating between grayscale and original color
        const new_r = gray + (rf - gray) * factor;
        const new_g = gray + (gf - gray) * factor;
        const new_b = gray + (bf - gray) * factor;

        pixel.r = @as(u8, @intFromFloat(std.math.clamp(new_r, 0.0, 255.0)));
        pixel.g = @as(u8, @intFromFloat(std.math.clamp(new_g, 0.0, 255.0)));
        pixel.b = @as(u8, @intFromFloat(std.math.clamp(new_b, 0.0, 255.0)));
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
        const tr = hk + 1.0 / 3.0;
        const tg = hk;
        const tb = hk - 1.0 / 3.0;

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

    // Convert degrees to radians
    const angle = hue_shift * std.math.pi / 180.0;
    const cos_a = @cos(angle);
    const sin_a = @sin(angle);

    // Rotation matrix for hue shift in RGB color space
    // Based on the standard color rotation matrix
    const sqrt3 = @sqrt(3.0);

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            const rf = @as(f32, @floatFromInt(pixels[idx].r));
            const gf = @as(f32, @floatFromInt(pixels[idx].g));
            const bf = @as(f32, @floatFromInt(pixels[idx].b));

            // Apply hue rotation matrix
            const new_r = rf * (cos_a + (1.0 - cos_a) / 3.0) +
                gf * ((1.0 - cos_a) / 3.0 - sin_a / sqrt3) +
                bf * ((1.0 - cos_a) / 3.0 + sin_a / sqrt3);

            const new_g = rf * ((1.0 - cos_a) / 3.0 + sin_a / sqrt3) +
                gf * (cos_a + (1.0 - cos_a) / 3.0) +
                bf * ((1.0 - cos_a) / 3.0 - sin_a / sqrt3);

            const new_b = rf * ((1.0 - cos_a) / 3.0 - sin_a / sqrt3) +
                gf * ((1.0 - cos_a) / 3.0 + sin_a / sqrt3) +
                bf * (cos_a + (1.0 - cos_a) / 3.0);

            pixels[idx].r = @as(u8, @intFromFloat(std.math.clamp(new_r, 0.0, 255.0)));
            pixels[idx].g = @as(u8, @intFromFloat(std.math.clamp(new_g, 0.0, 255.0)));
            pixels[idx].b = @as(u8, @intFromFloat(std.math.clamp(new_b, 0.0, 255.0)));
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

pub fn thresholdImage(ctx: *Context, args: anytype) !void {
    const threshold = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying threshold effect at {d}", .{threshold});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            const rf = @as(f32, @floatFromInt(pixels[idx].r));
            const gf = @as(f32, @floatFromInt(pixels[idx].g));
            const bf = @as(f32, @floatFromInt(pixels[idx].b));

            // Calculate luminance
            const luminance = 0.299 * rf + 0.587 * gf + 0.114 * bf;

            // Set to pure black or white based on threshold
            const value: u8 = if (luminance >= @as(f32, @floatFromInt(threshold))) 255 else 0;
            pixels[idx].r = value;
            pixels[idx].g = value;
            pixels[idx].b = value;
        }
    }
}

pub fn solarizeImage(ctx: *Context, args: anytype) !void {
    const threshold = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying solarize effect at threshold {d}", .{threshold});
    }

    const pixels = ctx.image.pixels.rgba32;

    for (pixels) |*pixel| {
        if (pixel.r > threshold) pixel.r = 255 - pixel.r;
        if (pixel.g > threshold) pixel.g = 255 - pixel.g;
        if (pixel.b > threshold) pixel.b = 255 - pixel.b;
    }
}

pub fn edgeDetectImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying edge detection (Sobel operator)", .{});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.allocator.alloc(img.color.Rgba32, width * height);
    defer ctx.allocator.free(temp_pixels);
    @memcpy(temp_pixels, pixels);

    // Sobel kernels
    const gx = [_][3]i32{
        .{ -1, 0, 1 },
        .{ -2, 0, 2 },
        .{ -1, 0, 1 },
    };
    const gy = [_][3]i32{
        .{ -1, -2, -1 },
        .{ 0, 0, 0 },
        .{ 1, 2, 1 },
    };

    // Process all pixels
    for (0..height) |y| {
        for (0..width) |x| {
            var sum_gx_r: i32 = 0;
            var sum_gx_g: i32 = 0;
            var sum_gx_b: i32 = 0;
            var sum_gy_r: i32 = 0;
            var sum_gy_g: i32 = 0;
            var sum_gy_b: i32 = 0;

            // Apply kernel
            for (0..3) |ky| {
                for (0..3) |kx| {
                    const px = @as(i32, @intCast(x)) + @as(i32, @intCast(kx)) - 1;
                    const py = @as(i32, @intCast(y)) + @as(i32, @intCast(ky)) - 1;

                    // Edge replication
                    const sample_x = std.math.clamp(px, 0, @as(i32, @intCast(width - 1)));
                    const sample_y = std.math.clamp(py, 0, @as(i32, @intCast(height - 1)));
                    const sample_idx = @as(usize, @intCast(sample_y * @as(i32, @intCast(width)) + sample_x));

                    const r = @as(i32, @intCast(temp_pixels[sample_idx].r));
                    const g = @as(i32, @intCast(temp_pixels[sample_idx].g));
                    const b = @as(i32, @intCast(temp_pixels[sample_idx].b));

                    sum_gx_r += r * gx[ky][kx];
                    sum_gx_g += g * gx[ky][kx];
                    sum_gx_b += b * gx[ky][kx];

                    sum_gy_r += r * gy[ky][kx];
                    sum_gy_g += g * gy[ky][kx];
                    sum_gy_b += b * gy[ky][kx];
                }
            }

            // Calculate magnitude
            const magnitude_r = @sqrt(@as(f32, @floatFromInt(sum_gx_r * sum_gx_r + sum_gy_r * sum_gy_r)));
            const magnitude_g = @sqrt(@as(f32, @floatFromInt(sum_gx_g * sum_gx_g + sum_gy_g * sum_gy_g)));
            const magnitude_b = @sqrt(@as(f32, @floatFromInt(sum_gx_b * sum_gx_b + sum_gy_b * sum_gy_b)));

            const idx = y * width + x;
            pixels[idx].r = @as(u8, @intFromFloat(std.math.clamp(magnitude_r, 0.0, 255.0)));
            pixels[idx].g = @as(u8, @intFromFloat(std.math.clamp(magnitude_g, 0.0, 255.0)));
            pixels[idx].b = @as(u8, @intFromFloat(std.math.clamp(magnitude_b, 0.0, 255.0)));
        }
    }
}

pub fn pixelateImage(ctx: *Context, args: anytype) !void {
    const block_size = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying pixelate effect with block size {d}", .{block_size});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Process image in blocks
    var by: usize = 0;
    while (by < height) : (by += block_size) {
        var bx: usize = 0;
        while (bx < width) : (bx += block_size) {
            // Calculate average color for this block
            var sum_r: u32 = 0;
            var sum_g: u32 = 0;
            var sum_b: u32 = 0;
            var count: u32 = 0;

            const block_end_y = @min(by + block_size, height);
            const block_end_x = @min(bx + block_size, width);

            for (by..block_end_y) |y| {
                for (bx..block_end_x) |x| {
                    const idx = y * width + x;
                    sum_r += pixels[idx].r;
                    sum_g += pixels[idx].g;
                    sum_b += pixels[idx].b;
                    count += 1;
                }
            }

            const avg_r = @as(u8, @intCast(sum_r / count));
            const avg_g = @as(u8, @intCast(sum_g / count));
            const avg_b = @as(u8, @intCast(sum_b / count));

            // Apply average color to entire block
            for (by..block_end_y) |y| {
                for (bx..block_end_x) |x| {
                    const idx = y * width + x;
                    pixels[idx].r = avg_r;
                    pixels[idx].g = avg_g;
                    pixels[idx].b = avg_b;
                }
            }
        }
    }
}

pub fn addNoiseImage(ctx: *Context, args: anytype) !void {
    const amount = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Adding noise with amount {d:.2}", .{amount});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Initialize random number generator
    var prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
    const rand = prng.random();

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;

            // Generate random noise in range [-amount * 255, amount * 255]
            const noise_r = (rand.float(f32) - 0.5) * 2.0 * amount * 255.0;
            const noise_g = (rand.float(f32) - 0.5) * 2.0 * amount * 255.0;
            const noise_b = (rand.float(f32) - 0.5) * 2.0 * amount * 255.0;

            const new_r = @as(f32, @floatFromInt(pixels[idx].r)) + noise_r;
            const new_g = @as(f32, @floatFromInt(pixels[idx].g)) + noise_g;
            const new_b = @as(f32, @floatFromInt(pixels[idx].b)) + noise_b;

            pixels[idx].r = @as(u8, @intFromFloat(std.math.clamp(new_r, 0.0, 255.0)));
            pixels[idx].g = @as(u8, @intFromFloat(std.math.clamp(new_g, 0.0, 255.0)));
            pixels[idx].b = @as(u8, @intFromFloat(std.math.clamp(new_b, 0.0, 255.0)));
        }
    }
}

pub fn adjustExposure(ctx: *Context, args: anytype) !void {
    const exposure = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying exposure adjustment by {d:.2}", .{exposure});
    }

    const pixels = ctx.image.pixels.rgba32;

    // Exposure uses exponential scaling (like camera EV)
    const multiplier = std.math.pow(f32, 2.0, exposure);

    for (pixels) |*pixel| {
        const rf = @as(f32, @floatFromInt(pixel.r)) * multiplier;
        const gf = @as(f32, @floatFromInt(pixel.g)) * multiplier;
        const bf = @as(f32, @floatFromInt(pixel.b)) * multiplier;

        pixel.r = @as(u8, @intFromFloat(std.math.clamp(rf, 0.0, 255.0)));
        pixel.g = @as(u8, @intFromFloat(std.math.clamp(gf, 0.0, 255.0)));
        pixel.b = @as(u8, @intFromFloat(std.math.clamp(bf, 0.0, 255.0)));
    }
}

pub fn adjustVibrance(ctx: *Context, args: anytype) !void {
    const factor = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying vibrance adjustment by {d:.2}", .{factor});
    }

    const pixels = ctx.image.pixels.rgba32;

    for (pixels) |*pixel| {
        const rf = @as(f32, @floatFromInt(pixel.r));
        const gf = @as(f32, @floatFromInt(pixel.g));
        const bf = @as(f32, @floatFromInt(pixel.b));

        // Calculate average
        const avg = (rf + gf + bf) / 3.0;

        // Calculate maximum saturation
        const max_val = @max(rf, @max(gf, bf));
        const min_val = @min(rf, @min(gf, bf));
        const sat = if (max_val > 0) (max_val - min_val) / max_val else 0.0;

        // Vibrance adjusts less saturated colors more (protects skin tones)
        const adjustment = factor * (1.0 - sat);

        const new_r = avg + (rf - avg) * (1.0 + adjustment);
        const new_g = avg + (gf - avg) * (1.0 + adjustment);
        const new_b = avg + (bf - avg) * (1.0 + adjustment);

        pixel.r = @as(u8, @intFromFloat(std.math.clamp(new_r, 0.0, 255.0)));
        pixel.g = @as(u8, @intFromFloat(std.math.clamp(new_g, 0.0, 255.0)));
        pixel.b = @as(u8, @intFromFloat(std.math.clamp(new_b, 0.0, 255.0)));
    }
}

pub fn equalizeImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying histogram equalization", .{});
    }

    const pixels = ctx.image.pixels.rgba32;
    const total_pixels = pixels.len;

    // Build histogram for luminance
    var histogram = [_]u32{0} ** 256;
    for (pixels) |pixel| {
        const rf = @as(f32, @floatFromInt(pixel.r));
        const gf = @as(f32, @floatFromInt(pixel.g));
        const bf = @as(f32, @floatFromInt(pixel.b));
        const luminance = 0.299 * rf + 0.587 * gf + 0.114 * bf;
        const lum_idx = @as(usize, @intFromFloat(std.math.clamp(luminance, 0.0, 255.0)));
        histogram[lum_idx] += 1;
    }

    // Build cumulative distribution function
    var cdf = [_]u32{0} ** 256;
    cdf[0] = histogram[0];
    for (1..256) |i| {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Find minimum non-zero CDF value
    var cdf_min: u32 = cdf[0];
    for (cdf) |value| {
        if (value > 0 and value < cdf_min) {
            cdf_min = value;
        }
    }

    // Create lookup table
    var lut = [_]u8{0} ** 256;
    const divisor = @as(f32, @floatFromInt(total_pixels - cdf_min));
    for (0..256) |i| {
        const numerator = @as(f32, @floatFromInt(cdf[i] - cdf_min));
        lut[i] = @as(u8, @intFromFloat(std.math.clamp((numerator / divisor) * 255.0, 0.0, 255.0)));
    }

    // Apply equalization
    for (pixels) |*pixel| {
        const rf = @as(f32, @floatFromInt(pixel.r));
        const gf = @as(f32, @floatFromInt(pixel.g));
        const bf = @as(f32, @floatFromInt(pixel.b));

        const old_lum = 0.299 * rf + 0.587 * gf + 0.114 * bf;
        const old_lum_idx = @as(usize, @intFromFloat(std.math.clamp(old_lum, 0.0, 255.0)));
        const new_lum = @as(f32, @floatFromInt(lut[old_lum_idx]));

        // Preserve color ratios while adjusting luminance
        const scale = if (old_lum > 0) new_lum / old_lum else 1.0;

        pixel.r = @as(u8, @intFromFloat(std.math.clamp(rf * scale, 0.0, 255.0)));
        pixel.g = @as(u8, @intFromFloat(std.math.clamp(gf * scale, 0.0, 255.0)));
        pixel.b = @as(u8, @intFromFloat(std.math.clamp(bf * scale, 0.0, 255.0)));
    }
}

pub fn colorizeImage(ctx: *Context, args: anytype) !void {
    const tint_r = args[0];
    const tint_g = args[1];
    const tint_b = args[2];
    const intensity = args[3];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying colorize with RGB({d}, {d}, {d}) intensity {d:.2}", .{ tint_r, tint_g, tint_b, intensity });
    }

    const pixels = ctx.image.pixels.rgba32;

    for (pixels) |*pixel| {
        const rf = @as(f32, @floatFromInt(pixel.r));
        const gf = @as(f32, @floatFromInt(pixel.g));
        const bf = @as(f32, @floatFromInt(pixel.b));

        // Calculate luminance
        const luminance = 0.299 * rf + 0.587 * gf + 0.114 * bf;

        // Blend with tint color based on luminance
        const tint_rf = @as(f32, @floatFromInt(tint_r));
        const tint_gf = @as(f32, @floatFromInt(tint_g));
        const tint_bf = @as(f32, @floatFromInt(tint_b));

        const tinted_r = (luminance / 255.0) * tint_rf;
        const tinted_g = (luminance / 255.0) * tint_gf;
        const tinted_b = (luminance / 255.0) * tint_bf;

        // Blend original with tinted
        const new_r = rf * (1.0 - intensity) + tinted_r * intensity;
        const new_g = gf * (1.0 - intensity) + tinted_g * intensity;
        const new_b = bf * (1.0 - intensity) + tinted_b * intensity;

        pixel.r = @as(u8, @intFromFloat(std.math.clamp(new_r, 0.0, 255.0)));
        pixel.g = @as(u8, @intFromFloat(std.math.clamp(new_g, 0.0, 255.0)));
        pixel.b = @as(u8, @intFromFloat(std.math.clamp(new_b, 0.0, 255.0)));
    }
}

pub fn duotoneImage(ctx: *Context, args: anytype) !void {
    const dark_r = args[0];
    const dark_g = args[1];
    const dark_b = args[2];
    const light_r = args[3];
    const light_g = args[4];
    const light_b = args[5];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying duotone effect", .{});
    }

    const pixels = ctx.image.pixels.rgba32;

    for (pixels) |*pixel| {
        const rf = @as(f32, @floatFromInt(pixel.r));
        const gf = @as(f32, @floatFromInt(pixel.g));
        const bf = @as(f32, @floatFromInt(pixel.b));

        // Calculate luminance (0-1 range)
        const luminance = (0.299 * rf + 0.587 * gf + 0.114 * bf) / 255.0;

        // Interpolate between dark and light colors based on luminance
        const dark_rf = @as(f32, @floatFromInt(dark_r));
        const dark_gf = @as(f32, @floatFromInt(dark_g));
        const dark_bf = @as(f32, @floatFromInt(dark_b));
        const light_rf = @as(f32, @floatFromInt(light_r));
        const light_gf = @as(f32, @floatFromInt(light_g));
        const light_bf = @as(f32, @floatFromInt(light_b));

        pixel.r = @as(u8, @intFromFloat(dark_rf + (light_rf - dark_rf) * luminance));
        pixel.g = @as(u8, @intFromFloat(dark_gf + (light_gf - dark_gf) * luminance));
        pixel.b = @as(u8, @intFromFloat(dark_bf + (light_bf - dark_bf) * luminance));
    }
}

pub fn oilPaintingImage(ctx: *Context, args: anytype) !void {
    const radius = args[0];
    try ctx.image.convert(ctx.allocator, .rgba32);

    if (ctx.verbose) {
        std.log.info("Applying oil painting effect with radius {d}", .{radius});
    }

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.allocator.alloc(img.color.Rgba32, width * height);
    defer ctx.allocator.free(temp_pixels);
    @memcpy(temp_pixels, pixels);

    // Oil painting uses intensity levels for a painted look
    const intensity_levels: usize = 20;

    for (0..height) |y| {
        for (0..width) |x| {
            // Count intensities in the neighborhood
            var intensity_count = [_]u32{0} ** 20;
            var avg_r = [_]f32{0} ** 20;
            var avg_g = [_]f32{0} ** 20;
            var avg_b = [_]f32{0} ** 20;

            // Scan neighborhood
            const y_start = if (y >= radius) y - radius else 0;
            const y_end = @min(y + radius + 1, height);
            const x_start = if (x >= radius) x - radius else 0;
            const x_end = @min(x + radius + 1, width);

            for (y_start..y_end) |ny| {
                for (x_start..x_end) |nx| {
                    const idx = ny * width + nx;
                    const rf = @as(f32, @floatFromInt(temp_pixels[idx].r));
                    const gf = @as(f32, @floatFromInt(temp_pixels[idx].g));
                    const bf = @as(f32, @floatFromInt(temp_pixels[idx].b));

                    // Calculate intensity level
                    const intensity = (rf + gf + bf) / 3.0;
                    const level = @min(@as(usize, @intFromFloat(intensity / 255.0 * @as(f32, @floatFromInt(intensity_levels - 1)))), intensity_levels - 1);

                    intensity_count[level] += 1;
                    avg_r[level] += rf;
                    avg_g[level] += gf;
                    avg_b[level] += bf;
                }
            }

            // Find most common intensity level
            var max_count: u32 = 0;
            var max_level: usize = 0;
            for (0..intensity_levels) |level| {
                if (intensity_count[level] > max_count) {
                    max_count = intensity_count[level];
                    max_level = level;
                }
            }

            // Set pixel to average color of most common intensity
            const idx = y * width + x;
            if (max_count > 0) {
                const count_f = @as(f32, @floatFromInt(max_count));
                pixels[idx].r = @as(u8, @intFromFloat(std.math.clamp(avg_r[max_level] / count_f, 0.0, 255.0)));
                pixels[idx].g = @as(u8, @intFromFloat(std.math.clamp(avg_g[max_level] / count_f, 0.0, 255.0)));
                pixels[idx].b = @as(u8, @intFromFloat(std.math.clamp(avg_b[max_level] / count_f, 0.0, 255.0)));
            }
        }
    }
}
