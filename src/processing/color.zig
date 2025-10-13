const img = @import("zigimg");
const std = @import("std");
const Context = @import("../core/types.zig").Context;
const utils = @import("../core/utils.zig");
const simd = @import("../utils/simd_utils.zig");
const math = std.math;

// Color Adjustments
pub fn adjustBrightness(ctx: *Context, args: anytype) !void {
    const delta = args[0];
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying brightness adjustment by {}", .{delta});

    const pixels = ctx.image.pixels.rgba32;
    simd.processPixelsSIMD4WithContext(pixels, delta, simd.adjustBrightnessPixel);
}

pub fn adjustContrast(ctx: *Context, args: anytype) !void {
    const factor = args[0];
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying contrast adjustment by {}", .{factor});

    const pixels = ctx.image.pixels.rgba32;
    simd.processPixelsSIMD4WithContext(pixels, factor, simd.adjustContrastPixel);
}

pub fn adjustSaturation(ctx: *Context, args: anytype) !void {
    const factor = args[0];
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying saturation adjustment by {}", .{factor});

    // Use SIMD processing with proper vector operations
    const pixels = ctx.image.pixels.rgba32;
    const pixel_count = pixels.len;
    var i: usize = 0;

    // Process in chunks of 4 pixels using actual SIMD vector operations
    while (i + 4 <= pixel_count) : (i += 4) {
        // Load 4 pixels and convert to f32 vectors
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

        // Calculate luminance for each pixel using SIMD
        const lum_vec = r_vec * @as(simd.Vec4f32, @splat(0.299)) +
            g_vec * @as(simd.Vec4f32, @splat(0.587)) +
            b_vec * @as(simd.Vec4f32, @splat(0.114));

        // Adjust saturation: new_color = lum + (color - lum) * factor
        const factor_vec = @as(simd.Vec4f32, @splat(factor));
        const new_r_vec = lum_vec + (r_vec - lum_vec) * factor_vec;
        const new_g_vec = lum_vec + (g_vec - lum_vec) * factor_vec;
        const new_b_vec = lum_vec + (b_vec - lum_vec) * factor_vec;

        // Clamp and convert back to u8
        const clamped_r = simd.clampVec4F32ToU8(new_r_vec);
        const clamped_g = simd.clampVec4F32ToU8(new_g_vec);
        const clamped_b = simd.clampVec4F32ToU8(new_b_vec);

        // Store back (preserve original alpha)
        pixels[i] = img.color.Rgba32{ .r = clamped_r[0], .g = clamped_g[0], .b = clamped_b[0], .a = pixels[i].a };
        pixels[i + 1] = img.color.Rgba32{ .r = clamped_r[1], .g = clamped_g[1], .b = clamped_b[1], .a = pixels[i + 1].a };
        pixels[i + 2] = img.color.Rgba32{ .r = clamped_r[2], .g = clamped_g[2], .b = clamped_b[2], .a = pixels[i + 2].a };
        pixels[i + 3] = img.color.Rgba32{ .r = clamped_r[3], .g = clamped_g[3], .b = clamped_b[3], .a = pixels[i + 3].a };
    }

    // Handle remaining pixels with scalar operations
    while (i < pixel_count) : (i += 1) {
        const pixel = &pixels[i];
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

pub fn adjustGamma(ctx: *Context, args: anytype) !void {
    const gamma = args[0];

    // Input validation with descriptive error messages
    if (gamma <= 0) {
        std.log.err("Gamma correction requires gamma > 0, got {d}", .{gamma});
        return error.InvalidGamma;
    }
    if (gamma > 10.0) {
        std.log.warn("Very high gamma value {d} may produce unexpected results", .{gamma});
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying gamma correction with gamma {}", .{gamma});

    const inv_gamma = 1.0 / gamma;
    const pixels = ctx.image.pixels.rgba32;
    simd.processPixelsSIMD4WithContext(pixels, inv_gamma, struct {
        fn process(pixel: img.color.Rgba32, inv_gamma_val: f32) img.color.Rgba32 {
            return img.color.Rgba32{
                .r = @as(u8, @intFromFloat(255.0 * std.math.pow(f32, @as(f32, @floatFromInt(pixel.r)) / 255.0, inv_gamma_val))),
                .g = @as(u8, @intFromFloat(255.0 * std.math.pow(f32, @as(f32, @floatFromInt(pixel.g)) / 255.0, inv_gamma_val))),
                .b = @as(u8, @intFromFloat(255.0 * std.math.pow(f32, @as(f32, @floatFromInt(pixel.b)) / 255.0, inv_gamma_val))),
                .a = pixel.a,
            };
        }
    }.process);
}

pub fn adjustExposure(ctx: *Context, args: anytype) !void {
    const exposure = args[0];
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying exposure adjustment by {d:.2}", .{exposure});

    const pixels = ctx.image.pixels.rgba32;

    // Exposure uses exponential scaling (like camera EV)
    const multiplier = std.math.pow(f32, 2.0, exposure);
    simd.processPixelsSIMD4WithContext(pixels, multiplier, struct {
        fn process(pixel: img.color.Rgba32, mult: f32) img.color.Rgba32 {
            const rf = @as(f32, @floatFromInt(pixel.r)) * mult;
            const gf = @as(f32, @floatFromInt(pixel.g)) * mult;
            const bf = @as(f32, @floatFromInt(pixel.b)) * mult;

            return img.color.Rgba32{
                .r = @as(u8, @intFromFloat(std.math.clamp(rf, 0.0, 255.0))),
                .g = @as(u8, @intFromFloat(std.math.clamp(gf, 0.0, 255.0))),
                .b = @as(u8, @intFromFloat(std.math.clamp(bf, 0.0, 255.0))),
                .a = pixel.a,
            };
        }
    }.process);
}

pub fn adjustVibrance(ctx: *Context, args: anytype) !void {
    const factor = args[0];
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying vibrance adjustment by {d:.2}", .{factor});

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
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying histogram equalization", .{});

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

pub fn hueShiftImage(ctx: *Context, args: anytype) !void {
    const hue_shift = args[0];
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying hue shift by {d:.2} degrees", .{hue_shift});

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Convert degrees to radians
    const angle = hue_shift * math.pi / 180.0;
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

// Color Effects
pub fn invertColors(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying invert modifier", .{});

    const pixels = ctx.image.pixels.rgba32;
    simd.processPixelsSIMD4(pixels, simd.invertPixel);
}

pub fn grayscaleImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying grayscale modifier", .{});

    const pixels = ctx.image.pixels.rgba32;
    simd.processPixelsSIMD4(pixels, simd.grayscalePixel);
}

pub fn applySepia(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying sepia tone effect", .{});

    const pixels = ctx.image.pixels.rgba32;
    simd.processPixelsSIMD4(pixels, simd.applySepiaPixel);
}

pub fn colorizeImage(ctx: *Context, args: anytype) !void {
    const tint_r = args[0];
    const tint_g = args[1];
    const tint_b = args[2];
    const intensity = args[3];

    // Input validation with descriptive error messages
    if (tint_r < 0 or tint_r > 255 or tint_g < 0 or tint_g > 255 or tint_b < 0 or tint_b > 255) {
        std.log.err("Colorize tint values must be in range 0-255, got RGB({d}, {d}, {d})", .{ tint_r, tint_g, tint_b });
        return error.InvalidColorValues;
    }
    if (intensity < 0.0 or intensity > 1.0) {
        std.log.err("Colorize intensity must be in range 0.0-1.0, got {d}", .{intensity});
        return error.InvalidIntensity;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying colorize with RGB({d}, {d}, {d}) intensity {d:.2}", .{ tint_r, tint_g, tint_b, intensity });

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

    // Input validation with descriptive error messages
    if (dark_r < 0 or dark_r > 255 or dark_g < 0 or dark_g > 255 or dark_b < 0 or dark_b > 255 or
        light_r < 0 or light_r > 255 or light_g < 0 or light_g > 255 or light_b < 0 or light_b > 255)
    {
        std.log.err("Duotone colors must be in range 0-255, got dark RGB({d},{d},{d}) light RGB({d},{d},{d})", .{ dark_r, dark_g, dark_b, light_r, light_g, light_b });
        return error.InvalidColorValues;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying duotone effect", .{});

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

pub fn thresholdImage(ctx: *Context, args: anytype) !void {
    const threshold = args[0];

    // Input validation
    if (threshold < 0 or threshold > 255) {
        return error.InvalidThreshold;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying threshold effect at {d}", .{threshold});

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            const luminance = utils.rgbToLuminanceU8(pixels[idx].r, pixels[idx].g, pixels[idx].b);

            // Set to pure black or white based on threshold
            const value: u8 = if (luminance >= threshold) 255 else 0;
            pixels[idx].r = value;
            pixels[idx].g = value;
            pixels[idx].b = value;
        }
    }
}

pub fn solarizeImage(ctx: *Context, args: anytype) !void {
    const threshold = args[0];

    // Input validation
    if (threshold < 0 or threshold > 255) {
        return error.InvalidThreshold;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying solarize effect at threshold {d}", .{threshold});

    const pixels = ctx.image.pixels.rgba32;

    for (pixels) |*pixel| {
        if (pixel.r > threshold) pixel.r = 255 - pixel.r;
        if (pixel.g > threshold) pixel.g = 255 - pixel.g;
        if (pixel.b > threshold) pixel.b = 255 - pixel.b;
    }
}

pub fn posterizeImage(ctx: *Context, args: anytype) !void {
    const levels = args[0];

    // Input validation
    if (levels <= 1) {
        return error.InvalidLevels;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying posterize effect with {d} levels", .{levels});

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
