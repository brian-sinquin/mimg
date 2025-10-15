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

    // Handle remaining pixels with scalar operations using inline helpers
    while (i < pixel_count) : (i += 1) {
        const pixel = &pixels[i];
        const rf = utils.u8ToF32(pixel.r);
        const gf = utils.u8ToF32(pixel.g);
        const bf = utils.u8ToF32(pixel.b);

        // Calculate luminance using inline helper
        const gray = utils.luminanceF32(rf, gf, bf);

        // Adjust saturation by interpolating between grayscale and original color
        const new_r = gray + (rf - gray) * factor;
        const new_g = gray + (gf - gray) * factor;
        const new_b = gray + (bf - gray) * factor;

        pixel.r = utils.clampU8(new_r);
        pixel.g = utils.clampU8(new_g);
        pixel.b = utils.clampU8(new_b);
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
    const pixel_count = pixels.len;
    var i: usize = 0;

    // Process in chunks of 4 pixels using SIMD
    while (i + 4 <= pixel_count) : (i += 4) {
        // Load 4 pixels into vectors
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

        // Apply vibrance adjustment using SIMD
        const result = simd.adjustVibranceSIMD4(r_vec, g_vec, b_vec, factor);
        const new_r_vec = simd.clampVec4F32ToU8(result[0]);
        const new_g_vec = simd.clampVec4F32ToU8(result[1]);
        const new_b_vec = simd.clampVec4F32ToU8(result[2]);

        // Store back (preserve alpha)
        pixels[i].r = new_r_vec[0];
        pixels[i].g = new_g_vec[0];
        pixels[i].b = new_b_vec[0];

        pixels[i + 1].r = new_r_vec[1];
        pixels[i + 1].g = new_g_vec[1];
        pixels[i + 1].b = new_b_vec[1];

        pixels[i + 2].r = new_r_vec[2];
        pixels[i + 2].g = new_g_vec[2];
        pixels[i + 2].b = new_b_vec[2];

        pixels[i + 3].r = new_r_vec[3];
        pixels[i + 3].g = new_g_vec[3];
        pixels[i + 3].b = new_b_vec[3];
    }

    // Handle remaining pixels with scalar operations
    while (i < pixel_count) : (i += 1) {
        const pixel = &pixels[i];
        const rf = utils.u8ToF32(pixel.r);
        const gf = utils.u8ToF32(pixel.g);
        const bf = utils.u8ToF32(pixel.b);

        // Calculate average using multiplication instead of division
        const avg = (rf + gf + bf) * utils.INV_3;

        // Calculate maximum saturation
        const max_val = @max(rf, @max(gf, bf));
        const min_val = @min(rf, @min(gf, bf));
        const sat = if (max_val > 0) (max_val - min_val) / max_val else 0.0;

        // Vibrance adjusts less saturated colors more (protects skin tones)
        const adjustment = factor * (1.0 - sat);

        const new_r = avg + (rf - avg) * (1.0 + adjustment);
        const new_g = avg + (gf - avg) * (1.0 + adjustment);
        const new_b = avg + (bf - avg) * (1.0 + adjustment);

        pixel.r = utils.clampU8(new_r);
        pixel.g = utils.clampU8(new_g);
        pixel.b = utils.clampU8(new_b);
    }
}

pub fn equalizeImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying histogram equalization", .{});

    const pixels = ctx.image.pixels.rgba32;
    const total_pixels = pixels.len;

    // Build histogram for luminance using inline helper
    var histogram = [_]u32{0} ** 256;
    for (pixels) |pixel| {
        const rf = utils.u8ToF32(pixel.r);
        const gf = utils.u8ToF32(pixel.g);
        const bf = utils.u8ToF32(pixel.b);
        const luminance = utils.luminanceF32(rf, gf, bf);
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

    // Pre-calculate repeated values (avoid repeated division in inner loop)
    const one_third = utils.INV_3;
    const one_minus_cos_third = (1.0 - cos_a) * one_third;
    const sin_div_sqrt3 = sin_a / utils.SQRT_3;
    const diagonal = cos_a + one_minus_cos_third;

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            const rf = utils.u8ToF32(pixels[idx].r);
            const gf = utils.u8ToF32(pixels[idx].g);
            const bf = utils.u8ToF32(pixels[idx].b);

            // Apply optimized hue rotation matrix (pre-calculated coefficients)
            const new_r = rf * diagonal +
                gf * (one_minus_cos_third - sin_div_sqrt3) +
                bf * (one_minus_cos_third + sin_div_sqrt3);

            const new_g = rf * (one_minus_cos_third + sin_div_sqrt3) +
                gf * diagonal +
                bf * (one_minus_cos_third - sin_div_sqrt3);

            const new_b = rf * (one_minus_cos_third - sin_div_sqrt3) +
                gf * (one_minus_cos_third + sin_div_sqrt3) +
                bf * diagonal;

            pixels[idx].r = utils.clampU8(new_r);
            pixels[idx].g = utils.clampU8(new_g);
            pixels[idx].b = utils.clampU8(new_b);
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
    const hex_color = args[0]; // e.g., "#RRGGBB" or "#RRGGBBAA"
    const intensity = args[1];

    // Input validation
    if (intensity < 0.0 or intensity > 1.0) {
        std.log.err("Colorize intensity must be in range 0.0-1.0, got {d}", .{intensity});
        return error.InvalidIntensity;
    }

    // Parse color
    const rgba = utils.parseHexColor(hex_color) catch {
        std.log.err("Invalid hex color '{s}'. Expected #RRGGBB or #RRGGBBAA.", .{hex_color});
        return error.InvalidHexColor;
    };

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying colorize with {s} (RGB {d},{d},{d}) intensity {d:.2}", .{ hex_color, rgba.r, rgba.g, rgba.b, intensity });

    const pixels = ctx.image.pixels.rgba32;

    for (pixels) |*pixel| {
        const rf = @as(f32, @floatFromInt(pixel.r));
        const gf = @as(f32, @floatFromInt(pixel.g));
        const bf = @as(f32, @floatFromInt(pixel.b));

        // Calculate luminance
        const luminance = 0.299 * rf + 0.587 * gf + 0.114 * bf;

        // Blend with tint color based on luminance
        const tint_rf = @as(f32, @floatFromInt(rgba.r));
        const tint_gf = @as(f32, @floatFromInt(rgba.g));
        const tint_bf = @as(f32, @floatFromInt(rgba.b));

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
    const dark_hex = args[0]; // "#RRGGBB" or "#RRGGBBAA"
    const light_hex = args[1]; // "#RRGGBB" or "#RRGGBBAA"

    const dark = utils.parseHexColor(dark_hex) catch {
        std.log.err("Invalid dark hex color '{s}'. Expected #RRGGBB or #RRGGBBAA.", .{dark_hex});
        return error.InvalidHexColor;
    };
    const light = utils.parseHexColor(light_hex) catch {
        std.log.err("Invalid light hex color '{s}'. Expected #RRGGBB or #RRGGBBAA.", .{light_hex});
        return error.InvalidHexColor;
    };

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying duotone effect from {s} to {s}", .{ dark_hex, light_hex });

    const pixels = ctx.image.pixels.rgba32;

    for (pixels) |*pixel| {
        const rf = @as(f32, @floatFromInt(pixel.r));
        const gf = @as(f32, @floatFromInt(pixel.g));
        const bf = @as(f32, @floatFromInt(pixel.b));

        // Calculate luminance (0-1 range)
        const luminance = (0.299 * rf + 0.587 * gf + 0.114 * bf) / 255.0;

        // Interpolate between dark and light colors based on luminance
        const dark_rf = @as(f32, @floatFromInt(dark.r));
        const dark_gf = @as(f32, @floatFromInt(dark.g));
        const dark_bf = @as(f32, @floatFromInt(dark.b));
        const light_rf = @as(f32, @floatFromInt(light.r));
        const light_gf = @as(f32, @floatFromInt(light.g));
        const light_bf = @as(f32, @floatFromInt(light.b));

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

pub fn adjustChannels(ctx: *Context, args: anytype) !void {
    const red_mult = args[0];
    const green_mult = args[1];
    const blue_mult = args[2];

    // Input validation
    if (red_mult < 0.0 or red_mult > 2.0 or green_mult < 0.0 or green_mult > 2.0 or blue_mult < 0.0 or blue_mult > 2.0) {
        std.log.err("Channel multipliers must be in range 0.0-2.0, got R:{d:.2} G:{d:.2} B:{d:.2}", .{ red_mult, green_mult, blue_mult });
        return error.InvalidMultiplier;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Adjusting channels - R:{d:.2} G:{d:.2} B:{d:.2}", .{ red_mult, green_mult, blue_mult });

    const pixels = ctx.image.pixels.rgba32;
    const multipliers = [_]f32{ red_mult, green_mult, blue_mult };

    // Use SIMD processing for channel adjustment
    const pixel_count = pixels.len;
    var i: usize = 0;

    // Process in chunks of 4 pixels
    while (i + 4 <= pixel_count) : (i += 4) {
        // Load 4 pixels
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

        // Apply multipliers
        const new_r_vec = r_vec * @as(simd.Vec4f32, @splat(multipliers[0]));
        const new_g_vec = g_vec * @as(simd.Vec4f32, @splat(multipliers[1]));
        const new_b_vec = b_vec * @as(simd.Vec4f32, @splat(multipliers[2]));

        // Clamp and convert back to u8
        const clamped_r = simd.clampVec4F32ToU8(new_r_vec);
        const clamped_g = simd.clampVec4F32ToU8(new_g_vec);
        const clamped_b = simd.clampVec4F32ToU8(new_b_vec);

        // Store back (preserve alpha)
        pixels[i] = img.color.Rgba32{ .r = clamped_r[0], .g = clamped_g[0], .b = clamped_b[0], .a = pixels[i].a };
        pixels[i + 1] = img.color.Rgba32{ .r = clamped_r[1], .g = clamped_g[1], .b = clamped_b[1], .a = pixels[i + 1].a };
        pixels[i + 2] = img.color.Rgba32{ .r = clamped_r[2], .g = clamped_g[2], .b = clamped_b[2], .a = pixels[i + 2].a };
        pixels[i + 3] = img.color.Rgba32{ .r = clamped_r[3], .g = clamped_g[3], .b = clamped_b[3], .a = pixels[i + 3].a };
    }

    // Handle remaining pixels
    while (i < pixel_count) : (i += 1) {
        const pixel = &pixels[i];
        const new_r = @as(f32, @floatFromInt(pixel.r)) * red_mult;
        const new_g = @as(f32, @floatFromInt(pixel.g)) * green_mult;
        const new_b = @as(f32, @floatFromInt(pixel.b)) * blue_mult;

        pixel.r = @as(u8, @intFromFloat(std.math.clamp(new_r, 0.0, 255.0)));
        pixel.g = @as(u8, @intFromFloat(std.math.clamp(new_g, 0.0, 255.0)));
        pixel.b = @as(u8, @intFromFloat(std.math.clamp(new_b, 0.0, 255.0)));
    }
}

pub fn adjustHSL(ctx: *Context, args: anytype) !void {
    const hue_shift = args[0];
    const saturation_factor = args[1];
    const lightness_factor = args[2];

    // Input validation
    if (hue_shift < -180.0 or hue_shift > 180.0) {
        std.log.err("Hue shift must be in range -180.0 to 180.0, got {d}", .{hue_shift});
        return error.InvalidHueShift;
    }
    if (saturation_factor < 0.0 or saturation_factor > 2.0) {
        std.log.err("Saturation factor must be in range 0.0-2.0, got {d}", .{saturation_factor});
        return error.InvalidSaturationFactor;
    }
    if (lightness_factor < 0.0 or lightness_factor > 2.0) {
        std.log.err("Lightness factor must be in range 0.0-2.0, got {d}", .{lightness_factor});
        return error.InvalidLightnessFactor;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Adjusting HSL - H:{d:.1} S:{d:.2} L:{d:.2}", .{ hue_shift, saturation_factor, lightness_factor });

    const pixels = ctx.image.pixels.rgba32;
    const pixel_count = pixels.len;

    // Process in chunks of 4 pixels using SIMD
    var i: usize = 0;
    while (i + 4 <= pixel_count) : (i += 4) {
        // Load 4 pixels
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

        // Apply SIMD HSL adjustment
        const result = simd.adjustHSL_SIMD4(r_vec, g_vec, b_vec, hue_shift, saturation_factor, lightness_factor);
        const new_r_vec = result[0];
        const new_g_vec = result[1];
        const new_b_vec = result[2];

        // Clamp and store back (preserve alpha)
        pixels[i].r = @as(u8, @intFromFloat(std.math.clamp(new_r_vec[0], 0.0, 255.0)));
        pixels[i].g = @as(u8, @intFromFloat(std.math.clamp(new_g_vec[0], 0.0, 255.0)));
        pixels[i].b = @as(u8, @intFromFloat(std.math.clamp(new_b_vec[0], 0.0, 255.0)));

        pixels[i + 1].r = @as(u8, @intFromFloat(std.math.clamp(new_r_vec[1], 0.0, 255.0)));
        pixels[i + 1].g = @as(u8, @intFromFloat(std.math.clamp(new_g_vec[1], 0.0, 255.0)));
        pixels[i + 1].b = @as(u8, @intFromFloat(std.math.clamp(new_b_vec[1], 0.0, 255.0)));

        pixels[i + 2].r = @as(u8, @intFromFloat(std.math.clamp(new_r_vec[2], 0.0, 255.0)));
        pixels[i + 2].g = @as(u8, @intFromFloat(std.math.clamp(new_g_vec[2], 0.0, 255.0)));
        pixels[i + 2].b = @as(u8, @intFromFloat(std.math.clamp(new_b_vec[2], 0.0, 255.0)));

        pixels[i + 3].r = @as(u8, @intFromFloat(std.math.clamp(new_r_vec[3], 0.0, 255.0)));
        pixels[i + 3].g = @as(u8, @intFromFloat(std.math.clamp(new_g_vec[3], 0.0, 255.0)));
        pixels[i + 3].b = @as(u8, @intFromFloat(std.math.clamp(new_b_vec[3], 0.0, 255.0)));
    }

    // Handle remaining pixels with scalar operations
    while (i < pixel_count) : (i += 1) {
        const pixel = &pixels[i];

        const rf = @as(f32, @floatFromInt(pixel.r)) / 255.0;
        const gf = @as(f32, @floatFromInt(pixel.g)) / 255.0;
        const bf = @as(f32, @floatFromInt(pixel.b)) / 255.0;

        // Convert RGB to HSL
        var h: f32 = 0.0;
        var s: f32 = 0.0;
        var l: f32 = 0.0;
        rgbToHsl(rf, gf, bf, &h, &s, &l);

        // Adjust HSL values
        h = @mod(h + hue_shift, 360.0);
        if (h < 0.0) h += 360.0;

        s = math.clamp(s * saturation_factor, 0.0, 1.0);
        l = math.clamp(l * lightness_factor, 0.0, 1.0);

        // Convert back to RGB
        var new_r: f32 = 0.0;
        var new_g: f32 = 0.0;
        var new_b: f32 = 0.0;
        hslToRgb(h, s, l, &new_r, &new_g, &new_b);

        pixel.r = @as(u8, @intFromFloat(math.clamp(new_r * 255.0, 0.0, 255.0)));
        pixel.g = @as(u8, @intFromFloat(math.clamp(new_g * 255.0, 0.0, 255.0)));
        pixel.b = @as(u8, @intFromFloat(math.clamp(new_b * 255.0, 0.0, 255.0)));
    }
}

/// Convert RGB (0-1) to HSL
fn rgbToHsl(r: f32, g: f32, b: f32, h: *f32, s: *f32, l: *f32) void {
    const max_val = @max(r, @max(g, b));
    const min_val = @min(r, @min(g, b));
    const delta = max_val - min_val;

    // Lightness
    l.* = (max_val + min_val) * utils.INV_2;

    // Saturation
    if (delta == 0.0) {
        s.* = 0.0;
        h.* = 0.0; // Undefined, but set to 0
    } else {
        s.* = if (l.* < 0.5) delta / (max_val + min_val) else delta / (2.0 - max_val - min_val);

        // Hue
        if (max_val == r) {
            h.* = (g - b) / delta;
        } else if (max_val == g) {
            h.* = 2.0 + (b - r) / delta;
        } else {
            h.* = 4.0 + (r - g) / delta;
        }
        h.* *= 60.0;
        if (h.* < 0.0) h.* += 360.0;
    }
}

/// Convert HSL to RGB (0-1)
fn hslToRgb(h: f32, s: f32, l: f32, r: *f32, g: *f32, b: *f32) void {
    if (s == 0.0) {
        // Achromatic (gray)
        r.* = l;
        g.* = l;
        b.* = l;
        return;
    }

    const hue = h / 360.0;
    const q = if (l < 0.5) l * (1.0 + s) else l + s - l * s;
    const p = 2.0 * l - q;

    r.* = hueToRgb(p, q, hue + 1.0 / 3.0);
    g.* = hueToRgb(p, q, hue);
    b.* = hueToRgb(p, q, hue - 1.0 / 3.0);
}

fn hueToRgb(p: f32, q: f32, t: f32) f32 {
    var t_clamped = t;
    if (t_clamped < 0.0) t_clamped += 1.0;
    if (t_clamped > 1.0) t_clamped -= 1.0;

    const INV_6: f32 = 1.0 / 6.0;
    const TWO_THIRDS: f32 = 2.0 / 3.0;

    if (t_clamped < INV_6) return p + (q - p) * 6.0 * t_clamped;
    if (t_clamped < utils.INV_2) return q;
    if (t_clamped < TWO_THIRDS) return p + (q - p) * (TWO_THIRDS - t_clamped) * 6.0;
    return p;
}

pub fn equalizeAreaImage(ctx: *Context, args: anytype) !void {
    const x = args[0];
    const y = args[1];
    const width = args[2];
    const height = args[3];

    // Input validation
    if (x < 0 or y < 0 or width <= 0 or height <= 0) {
        std.log.err("Area coordinates must be valid: x={}, y={}, w={}, h={}", .{ x, y, width, height });
        return error.InvalidArea;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying local histogram equalization to area {}x{} at ({}, {})", .{ width, height, x, y });
    utils.logMemoryUsage(ctx, "Equalize area start");

    const pixels = ctx.image.pixels.rgba32;
    const img_width = ctx.image.width;
    const img_height = ctx.image.height;

    // Clamp area to image bounds
    const area_x = @as(usize, @intCast(math.clamp(x, 0, @as(i32, @intCast(img_width - 1)))));
    const area_y = @as(usize, @intCast(math.clamp(y, 0, @as(i32, @intCast(img_height - 1)))));
    const area_width = @as(usize, @intCast(math.clamp(width, 1, @as(i32, @intCast(img_width - area_x)))));
    const area_height = @as(usize, @intCast(math.clamp(height, 1, @as(i32, @intCast(img_height - area_y)))));

    // Build histogram for luminance in the specified area
    var histogram = [_]u32{0} ** 256;
    for (area_y..area_y + area_height) |ay| {
        for (area_x..area_x + area_width) |ax| {
            const idx = ay * img_width + ax;
            const pixel = pixels[idx];

            const rf = @as(f32, @floatFromInt(pixel.r));
            const gf = @as(f32, @floatFromInt(pixel.g));
            const bf = @as(f32, @floatFromInt(pixel.b));
            const luminance = 0.299 * rf + 0.587 * gf + 0.114 * bf;
            const lum_idx = @as(usize, @intFromFloat(std.math.clamp(luminance, 0.0, 255.0)));
            histogram[lum_idx] += 1;
        }
    }

    // Build cumulative distribution function for the area
    var cdf = [_]u32{0} ** 256;
    cdf[0] = histogram[0];
    for (1..256) |i| {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Find minimum non-zero CDF value in the area
    var cdf_min: u32 = cdf[0];
    for (cdf) |value| {
        if (value > 0 and value < cdf_min) {
            cdf_min = value;
        }
    }

    // Create lookup table for equalization
    var lut = [_]u8{0} ** 256;
    const divisor = @as(f32, @floatFromInt(area_width * area_height - cdf_min));
    for (0..256) |i| {
        const numerator = @as(f32, @floatFromInt(cdf[i] - cdf_min));
        lut[i] = @as(u8, @intFromFloat(std.math.clamp((numerator / divisor) * 255.0, 0.0, 255.0)));
    }

    // Apply equalization to the specified area
    for (area_y..area_y + area_height) |ay| {
        for (area_x..area_x + area_width) |ax| {
            const idx = ay * img_width + ax;

            const pixel = pixels[idx];
            const rf = @as(f32, @floatFromInt(pixel.r));
            const gf = @as(f32, @floatFromInt(pixel.g));
            const bf = @as(f32, @floatFromInt(pixel.b));

            const old_lum = 0.299 * rf + 0.587 * gf + 0.114 * bf;
            const old_lum_idx = @as(usize, @intFromFloat(std.math.clamp(old_lum, 0.0, 255.0)));
            const new_lum = @as(f32, @floatFromInt(lut[old_lum_idx]));

            // Preserve color ratios while adjusting luminance
            const scale = if (old_lum > 0) new_lum / old_lum else 1.0;

            pixels[idx].r = @as(u8, @intFromFloat(std.math.clamp(rf * scale, 0.0, 255.0)));
            pixels[idx].g = @as(u8, @intFromFloat(std.math.clamp(gf * scale, 0.0, 255.0)));
            pixels[idx].b = @as(u8, @intFromFloat(std.math.clamp(bf * scale, 0.0, 255.0)));
        }
    }

    utils.logMemoryUsage(ctx, "Equalize area end");
}
