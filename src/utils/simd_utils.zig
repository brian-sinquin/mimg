const std = @import("std");
const img = @import("zigimg");

/// SIMD vector types for common operations
pub const Vec4f32 = @Vector(4, f32);
pub const Vec4u8 = @Vector(4, u8);

/// Common SIMD constants
pub const ZERO_F32: Vec4f32 = @splat(0.0);
pub const ONE_F32: Vec4f32 = @splat(1.0);
pub const MAX_U8_F32: Vec4f32 = @splat(255.0);

/// Clamp a float value to u8 range (0-255)
pub fn clampF32ToU8(value: f32) u8 {
    return @as(u8, @intFromFloat(std.math.clamp(value, 0.0, 255.0)));
}

/// Clamp a vector of f32 values to u8 range
pub fn clampVec4F32ToU8(vec: Vec4f32) Vec4u8 {
    const clamped = @max(ZERO_F32, @min(vec, MAX_U8_F32));
    return @as(Vec4u8, @intFromFloat(clamped));
}

/// Apply a function to each pixel in SIMD chunks of 4
pub fn processPixelsSIMD4(
    pixels: []img.color.Rgba32,
    comptime processFn: fn (img.color.Rgba32) img.color.Rgba32,
) void {
    var i: usize = 0;
    const pixel_count = pixels.len;

    // Process in chunks of 4 pixels
    while (i + 4 <= pixel_count) : (i += 4) {
        pixels[i] = processFn(pixels[i]);
        pixels[i + 1] = processFn(pixels[i + 1]);
        pixels[i + 2] = processFn(pixels[i + 2]);
        pixels[i + 3] = processFn(pixels[i + 3]);
    }

    // Handle remaining pixels with scalar operations
    while (i < pixel_count) : (i += 1) {
        pixels[i] = processFn(pixels[i]);
    }
}

/// Apply a function with context to each pixel in SIMD chunks of 4
pub fn processPixelsSIMD4WithContext(
    pixels: []img.color.Rgba32,
    context: anytype,
    comptime processFn: anytype,
) void {
    var i: usize = 0;
    const pixel_count = pixels.len;

    // Process in chunks of 4 pixels
    while (i + 4 <= pixel_count) : (i += 4) {
        pixels[i] = @call(.auto, processFn, .{ pixels[i], context });
        pixels[i + 1] = @call(.auto, processFn, .{ pixels[i + 1], context });
        pixels[i + 2] = @call(.auto, processFn, .{ pixels[i + 2], context });
        pixels[i + 3] = @call(.auto, processFn, .{ pixels[i + 3], context });
    }

    // Handle remaining pixels with scalar operations
    while (i < pixel_count) : (i += 1) {
        pixels[i] = @call(.auto, processFn, .{ pixels[i], context });
    }
}

/// Calculate luminance for a single pixel
pub fn calculateLuminance(pixel: img.color.Rgba32) f32 {
    const rf = @as(f32, @floatFromInt(pixel.r));
    const gf = @as(f32, @floatFromInt(pixel.g));
    const bf = @as(f32, @floatFromInt(pixel.b));
    return 0.299 * rf + 0.587 * gf + 0.114 * bf;
}

/// Apply brightness adjustment to a single pixel
pub fn adjustBrightnessPixel(pixel: img.color.Rgba32, delta: anytype) img.color.Rgba32 {
    return img.color.Rgba32{
        .r = @as(u8, @intCast(std.math.clamp(@as(i16, @intCast(pixel.r)) + delta, 0, 255))),
        .g = @as(u8, @intCast(std.math.clamp(@as(i16, @intCast(pixel.g)) + delta, 0, 255))),
        .b = @as(u8, @intCast(std.math.clamp(@as(i16, @intCast(pixel.b)) + delta, 0, 255))),
        .a = pixel.a,
    };
}

/// Apply contrast adjustment to a single pixel
pub fn adjustContrastPixel(pixel: img.color.Rgba32, factor: anytype) img.color.Rgba32 {
    return img.color.Rgba32{
        .r = @as(u8, @intFromFloat(std.math.clamp((@as(f32, @floatFromInt(pixel.r)) - 128.0) * factor + 128.0, 0.0, 255.0))),
        .g = @as(u8, @intFromFloat(std.math.clamp((@as(f32, @floatFromInt(pixel.g)) - 128.0) * factor + 128.0, 0.0, 255.0))),
        .b = @as(u8, @intFromFloat(std.math.clamp((@as(f32, @floatFromInt(pixel.b)) - 128.0) * factor + 128.0, 0.0, 255.0))),
        .a = pixel.a,
    };
}

/// Apply gamma correction to a single pixel
pub fn adjustGammaPixel(pixel: img.color.Rgba32, inv_gamma: f32) img.color.Rgba32 {
    return img.color.Rgba32{
        .r = @as(u8, @intFromFloat(255.0 * std.math.pow(f32, @as(f32, @floatFromInt(pixel.r)) / 255.0, inv_gamma))),
        .g = @as(u8, @intFromFloat(255.0 * std.math.pow(f32, @as(f32, @floatFromInt(pixel.g)) / 255.0, inv_gamma))),
        .b = @as(u8, @intFromFloat(255.0 * std.math.pow(f32, @as(f32, @floatFromInt(pixel.b)) / 255.0, inv_gamma))),
        .a = pixel.a,
    };
}

/// Apply sepia tone to a single pixel
pub fn applySepiaPixel(pixel: img.color.Rgba32) img.color.Rgba32 {
    const r = @as(f32, @floatFromInt(pixel.r));
    const g = @as(f32, @floatFromInt(pixel.g));
    const b = @as(f32, @floatFromInt(pixel.b));

    return img.color.Rgba32{
        .r = @as(u8, @intFromFloat(std.math.clamp(0.393 * r + 0.769 * g + 0.189 * b, 0.0, 255.0))),
        .g = @as(u8, @intFromFloat(std.math.clamp(0.349 * r + 0.686 * g + 0.168 * b, 0.0, 255.0))),
        .b = @as(u8, @intFromFloat(std.math.clamp(0.272 * r + 0.534 * g + 0.131 * b, 0.0, 255.0))),
        .a = pixel.a,
    };
}

/// Invert colors of a single pixel
pub fn invertPixel(pixel: img.color.Rgba32) img.color.Rgba32 {
    return img.color.Rgba32{
        .r = 255 - pixel.r,
        .g = 255 - pixel.g,
        .b = 255 - pixel.b,
        .a = pixel.a,
    };
}

/// Convert pixel to grayscale
pub fn grayscalePixel(pixel: img.color.Rgba32) img.color.Rgba32 {
    const gray = @import("../core/utils.zig").rgbToLuminanceU8(pixel.r, pixel.g, pixel.b);
    return img.color.Rgba32{
        .r = gray,
        .g = gray,
        .b = gray,
        .a = pixel.a,
    };
}

/// Calculate square root for each element in a Vec4f32
pub fn sqrtVec4F32(vec: Vec4f32) Vec4f32 {
    return [_]f32{
        @sqrt(vec[0]),
        @sqrt(vec[1]),
        @sqrt(vec[2]),
        @sqrt(vec[3]),
    };
}

/// Clamp a Vec4f32 to min/max values
pub fn clampVec4F32(vec: Vec4f32, min_val: Vec4f32, max_val: Vec4f32) Vec4f32 {
    return @max(min_val, @min(vec, max_val));
}

/// Apply vibrance adjustment to 4 pixels at once using SIMD
pub fn adjustVibranceSIMD4(r_vec: Vec4f32, g_vec: Vec4f32, b_vec: Vec4f32, factor: f32) struct { Vec4f32, Vec4f32, Vec4f32 } {
    // Calculate averages: (r + g + b) / 3
    const avg_vec = (r_vec + g_vec + b_vec) / @as(Vec4f32, @splat(3.0));

    // Find min and max values for each pixel
    const min_vec = @min(r_vec, @min(g_vec, b_vec));
    const max_vec = @max(r_vec, @max(g_vec, b_vec));

    // Calculate saturation: (max - min) / max, but handle max = 0 case
    const delta_vec = max_vec - min_vec;
    const sat_vec = @select(f32, max_vec > ZERO_F32, delta_vec / max_vec, ZERO_F32);

    // Calculate adjustment: factor * (1.0 - saturation)
    const factor_vec = @as(Vec4f32, @splat(factor));
    const adjustment_vec = factor_vec * (ONE_F32 - sat_vec);

    // Apply vibrance: avg + (original - avg) * (1.0 + adjustment)
    const scale_vec = ONE_F32 + adjustment_vec;
    const new_r_vec = avg_vec + (r_vec - avg_vec) * scale_vec;
    const new_g_vec = avg_vec + (g_vec - avg_vec) * scale_vec;
    const new_b_vec = avg_vec + (b_vec - avg_vec) * scale_vec;

    return .{ new_r_vec, new_g_vec, new_b_vec };
}

/// Apply vignette effect to 4 pixels at once using SIMD
pub fn applyVignetteSIMD4(
    r_vec: Vec4f32,
    g_vec: Vec4f32,
    b_vec: Vec4f32,
    x_vec: Vec4f32,
    y_vec: Vec4f32,
    center_x: f32,
    center_y: f32,
    inv_max_distance: f32,
    intensity: f32,
) struct { Vec4f32, Vec4f32, Vec4f32 } {
    // Calculate dx and dy
    const cx_vec = @as(Vec4f32, @splat(center_x));
    const cy_vec = @as(Vec4f32, @splat(center_y));
    const dx_vec = x_vec - cx_vec;
    const dy_vec = y_vec - cy_vec;

    // Calculate distance: sqrt(dx*dx + dy*dy)
    const dx_sq_vec = dx_vec * dx_vec;
    const dy_sq_vec = dy_vec * dy_vec;
    const dist_sq_vec = dx_sq_vec + dy_sq_vec;
    const dist_vec = sqrtVec4F32(dist_sq_vec);

    // Calculate normalized distance
    const inv_max_dist_vec = @as(Vec4f32, @splat(inv_max_distance));
    const normalized_dist_vec = dist_vec * inv_max_dist_vec;

    // Calculate vignette factor: 1.0 - (normalized_distance * intensity)
    const intensity_vec = @as(Vec4f32, @splat(intensity));
    const vignette_factor_vec = ONE_F32 - (normalized_dist_vec * intensity_vec);

    // Apply vignette: multiply each channel by vignette factor
    const new_r_vec = r_vec * vignette_factor_vec;
    const new_g_vec = g_vec * vignette_factor_vec;
    const new_b_vec = b_vec * vignette_factor_vec;

    return .{ new_r_vec, new_g_vec, new_b_vec };
}

/// Apply round corners effect to 4 pixels at once using SIMD with SDF approach
pub fn applyRoundCornersSIMD4(
    r_vec: Vec4f32,
    g_vec: Vec4f32,
    b_vec: Vec4f32,
    a_vec: Vec4f32,
    x_vec: Vec4f32,
    y_vec: Vec4f32,
    width: f32,
    height: f32,
    radius: f32,
) struct { Vec4f32, Vec4f32, Vec4f32, Vec4f32 } {
    // Calculate alpha using signed distance field for perfect anti-aliasing
    const alpha_mult_vec = calculateRoundedRectSDF_SIMD4(x_vec, y_vec, width, height, radius);

    // Apply alpha multiplier: new_alpha = current_alpha * alpha_mult
    const new_a_vec = a_vec * alpha_mult_vec;

    return .{ r_vec, g_vec, b_vec, new_a_vec };
}

// SIMD version of signed distance field function for rounded rectangle
fn calculateRoundedRectSDF_SIMD4(x_vec: Vec4f32, y_vec: Vec4f32, width: f32, height: f32, radius: f32) Vec4f32 {
    const width_vec = @as(Vec4f32, @splat(width));
    const height_vec = @as(Vec4f32, @splat(height));
    const radius_vec = @as(Vec4f32, @splat(radius));

    // Convert to center-based coordinates
    const cx_vec = x_vec - (width_vec - @as(Vec4f32, @splat(1.0))) / @as(Vec4f32, @splat(2.0));
    const cy_vec = y_vec - (height_vec - @as(Vec4f32, @splat(1.0))) / @as(Vec4f32, @splat(2.0));

    // Half dimensions
    const half_w_vec = (width_vec - @as(Vec4f32, @splat(1.0))) / @as(Vec4f32, @splat(2.0));
    const half_h_vec = (height_vec - @as(Vec4f32, @splat(1.0))) / @as(Vec4f32, @splat(2.0));

    // Calculate distance to rounded rectangle using SDF
    const abs_cx = @abs(cx_vec);
    const abs_cy = @abs(cy_vec);
    const dx_vec = @max(@as(Vec4f32, @splat(0.0)), abs_cx - (half_w_vec - radius_vec));
    const dy_vec = @max(@as(Vec4f32, @splat(0.0)), abs_cy - (half_h_vec - radius_vec));
    const distance_vec = sqrtVec4F32(dx_vec * dx_vec + dy_vec * dy_vec) - radius_vec;

    // Convert distance to alpha with smooth anti-aliasing
    const aa_width_vec = @as(Vec4f32, @splat(1.0)); // Anti-aliasing width in pixels

    // Create masks for different regions
    const outside_mask = distance_vec > aa_width_vec;
    const positive_dist_mask = distance_vec > @as(Vec4f32, @splat(0.0));
    const within_aa_mask = distance_vec <= aa_width_vec;
    const transition_mask = @select(f32, positive_dist_mask, @select(f32, within_aa_mask, @as(Vec4f32, @splat(1.0)), @as(Vec4f32, @splat(0.0))), @as(Vec4f32, @splat(0.0)));

    // Calculate alpha values
    const alpha_outside = @as(Vec4f32, @splat(0.0));
    const alpha_inside = @as(Vec4f32, @splat(1.0));

    // Smooth transition zone using smoothstep
    const t_vec = @as(Vec4f32, @splat(1.0)) - (distance_vec / aa_width_vec);
    const t_clamped = @max(@as(Vec4f32, @splat(0.0)), @min(@as(Vec4f32, @splat(1.0)), t_vec));
    const alpha_transition = t_clamped * t_clamped * (@as(Vec4f32, @splat(3.0)) - @as(Vec4f32, @splat(2.0)) * t_clamped);

    // Select appropriate alpha based on region
    return @select(f32, outside_mask, alpha_outside, @select(f32, transition_mask > @as(Vec4f32, @splat(0.0)), alpha_transition, alpha_inside));
}

/// Apply HSL adjustment to 4 pixels at once using SIMD
pub fn adjustHSL_SIMD4(
    r_vec: Vec4f32,
    g_vec: Vec4f32,
    b_vec: Vec4f32,
    hue_shift: f32,
    saturation_factor: f32,
    lightness_factor: f32,
) struct { Vec4f32, Vec4f32, Vec4f32 } {
    // Normalize RGB to 0-1 range
    const r_norm = r_vec / @as(Vec4f32, @splat(255.0));
    const g_norm = g_vec / @as(Vec4f32, @splat(255.0));
    const b_norm = b_vec / @as(Vec4f32, @splat(255.0));

    // Find max and min values for each pixel
    const max_vals = @max(r_norm, @max(g_norm, b_norm));
    const min_vals = @min(r_norm, @min(g_norm, b_norm));
    const deltas = max_vals - min_vals;

    // Calculate lightness
    const lightness = (max_vals + min_vals) / @as(Vec4f32, @splat(2.0));

    // Calculate saturation
    const sat_numerator = deltas;
    const sat_denominator = @select(f32, lightness < @as(Vec4f32, @splat(0.5)), max_vals + min_vals, @as(Vec4f32, @splat(2.0)) - max_vals - min_vals);
    const saturation = @select(f32, deltas == @as(Vec4f32, @splat(0.0)), @as(Vec4f32, @splat(0.0)), sat_numerator / sat_denominator);

    // Calculate hue
    const r_is_max = max_vals == r_norm;
    const g_is_max = max_vals == g_norm;
    const b_is_max = max_vals == b_norm;

    const hue_r = @select(f32, r_is_max, (g_norm - b_norm) / deltas, @as(Vec4f32, @splat(0.0)));
    const hue_g = @select(f32, g_is_max, @as(Vec4f32, @splat(2.0)) + (b_norm - r_norm) / deltas, @as(Vec4f32, @splat(0.0)));
    const hue_b = @select(f32, b_is_max, @as(Vec4f32, @splat(4.0)) + (r_norm - g_norm) / deltas, @as(Vec4f32, @splat(0.0)));

    var hue = (hue_r + hue_g + hue_b) * @as(Vec4f32, @splat(60.0));
    hue = @select(f32, hue < @as(Vec4f32, @splat(0.0)), hue + @as(Vec4f32, @splat(360.0)), hue);

    // Apply adjustments
    const hue_shift_vec = @as(Vec4f32, @splat(hue_shift));
    var adjusted_hue = @mod(hue + hue_shift_vec, @as(Vec4f32, @splat(360.0)));
    adjusted_hue = @select(f32, adjusted_hue < @as(Vec4f32, @splat(0.0)), adjusted_hue + @as(Vec4f32, @splat(360.0)), adjusted_hue);

    const sat_factor_vec = @as(Vec4f32, @splat(saturation_factor));
    const adjusted_sat = @min(saturation * sat_factor_vec, @as(Vec4f32, @splat(1.0)));

    const light_factor_vec = @as(Vec4f32, @splat(lightness_factor));
    const adjusted_light = @min(lightness * light_factor_vec, @as(Vec4f32, @splat(1.0)));

    // Convert back to RGB
    const hue_norm = adjusted_hue / @as(Vec4f32, @splat(360.0));

    // Achromatic case (saturation == 0)
    const achromatic_mask = adjusted_sat == @as(Vec4f32, @splat(0.0));
    const achromatic_rgb = @select(f32, achromatic_mask, adjusted_light, @as(Vec4f32, @splat(0.0)));

    // Chromatic case
    const q = @select(f32, adjusted_light < @as(Vec4f32, @splat(0.5)), adjusted_light * (@as(Vec4f32, @splat(1.0)) + adjusted_sat), adjusted_light + adjusted_sat - adjusted_light * adjusted_sat);
    const p = @as(Vec4f32, @splat(2.0)) * adjusted_light - q;

    const hue_third = @as(Vec4f32, @splat(1.0 / 3.0));
    const r_hue = hue_norm + hue_third;
    const g_hue = hue_norm;
    const b_hue = hue_norm - hue_third;

    const r_chromatic = hueToRgbVec4(p, q, r_hue);
    const g_chromatic = hueToRgbVec4(p, q, g_hue);
    const b_chromatic = hueToRgbVec4(p, q, b_hue);

    // Select between achromatic and chromatic results
    const final_r = @select(f32, achromatic_mask, achromatic_rgb, r_chromatic);
    const final_g = @select(f32, achromatic_mask, achromatic_rgb, g_chromatic);
    const final_b = @select(f32, achromatic_mask, achromatic_rgb, b_chromatic);

    // Convert back to 0-255 range
    return .{
        final_r * @as(Vec4f32, @splat(255.0)),
        final_g * @as(Vec4f32, @splat(255.0)),
        final_b * @as(Vec4f32, @splat(255.0)),
    };
}

/// Vectorized hue to RGB conversion helper
fn hueToRgbVec4(p: Vec4f32, q: Vec4f32, t: Vec4f32) Vec4f32 {
    var t_clamped = t;
    const one_vec = @as(Vec4f32, @splat(1.0));

    // Clamp t to [0, 1]
    t_clamped = @select(f32, t_clamped < @as(Vec4f32, @splat(0.0)), t_clamped + one_vec, t_clamped);
    t_clamped = @select(f32, t_clamped > one_vec, t_clamped - one_vec, t_clamped);

    const sixth = @as(Vec4f32, @splat(1.0 / 6.0));
    const half = @as(Vec4f32, @splat(0.5));
    const two_thirds = @as(Vec4f32, @splat(2.0 / 3.0));

    // t < 1/6
    const case1_mask = t_clamped < sixth;
    const case1_result = p + (q - p) * t_clamped * @as(Vec4f32, @splat(6.0));

    // 1/6 <= t < 1/2
    const case2_min = t_clamped >= sixth;
    const case2_max = t_clamped < half;
    const case2_mask = @select(f32, case2_min, @select(f32, case2_max, @as(Vec4f32, @splat(1.0)), @as(Vec4f32, @splat(0.0))), @as(Vec4f32, @splat(0.0)));
    const case2_result = @select(f32, case2_mask > @as(Vec4f32, @splat(0.0)), q, @as(Vec4f32, @splat(0.0)));

    // 1/2 <= t < 2/3
    const case3_min = t_clamped >= half;
    const case3_max = t_clamped < two_thirds;
    const case3_mask = @select(f32, case3_min, @select(f32, case3_max, @as(Vec4f32, @splat(1.0)), @as(Vec4f32, @splat(0.0))), @as(Vec4f32, @splat(0.0)));
    const case3_result = p + (q - p) * (two_thirds - t_clamped) * @as(Vec4f32, @splat(6.0));

    // t >= 2/3 (default case)
    const case4_result = p;

    // Combine results
    var result = case4_result;
    result = @select(f32, case1_mask, case1_result, result);
    result = @select(f32, case2_mask > @as(Vec4f32, @splat(0.0)), case2_result, result);
    result = @select(f32, case3_mask > @as(Vec4f32, @splat(0.0)), case3_result, result);

    return result;
}

/// Apply horizontal 1D convolution to 4 pixels at once using SIMD
/// This processes 4 consecutive pixels horizontally, handling boundary conditions
pub fn applyHorizontalConvolutionSIMD4(
    src_pixels: []const img.color.Rgba32,
    dst_pixels: []img.color.Rgba32,
    width: usize,
    kernel: []const f32,
    kernel_radius: usize,
    start_y: usize,
    start_x: usize,
) void {
    const kernel_size = kernel_radius * 2 + 1;

    // Process 4 pixels horizontally if they fit
    if (start_x + 4 > width) return; // Not enough pixels for SIMD

    // Load 4 consecutive pixels and their neighbors
    var r_sum_vec: Vec4f32 = @splat(0.0);
    var g_sum_vec: Vec4f32 = @splat(0.0);
    var b_sum_vec: Vec4f32 = @splat(0.0);
    var a_sum_vec: Vec4f32 = @splat(0.0);

    // Apply kernel to each of the 4 pixels
    for (0..kernel_size) |k| {
        const offset = @as(i32, @intCast(k)) - @as(i32, @intCast(kernel_radius));

        // Calculate source positions for each of the 4 pixels
        const src_x0 = @as(i32, @intCast(start_x)) + offset;
        const src_x1 = @as(i32, @intCast(start_x + 1)) + offset;
        const src_x2 = @as(i32, @intCast(start_x + 2)) + offset;
        const src_x3 = @as(i32, @intCast(start_x + 3)) + offset;

        // Clamp to image boundaries
        const clamped_x0 = std.math.clamp(src_x0, 0, @as(i32, @intCast(width - 1)));
        const clamped_x1 = std.math.clamp(src_x1, 0, @as(i32, @intCast(width - 1)));
        const clamped_x2 = std.math.clamp(src_x2, 0, @as(i32, @intCast(width - 1)));
        const clamped_x3 = std.math.clamp(src_x3, 0, @as(i32, @intCast(width - 1)));

        // Load pixel values
        const idx0 = @as(usize, @intCast(start_y)) * width + @as(usize, @intCast(clamped_x0));
        const idx1 = @as(usize, @intCast(start_y)) * width + @as(usize, @intCast(clamped_x1));
        const idx2 = @as(usize, @intCast(start_y)) * width + @as(usize, @intCast(clamped_x2));
        const idx3 = @as(usize, @intCast(start_y)) * width + @as(usize, @intCast(clamped_x3));

        const r_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].r)),
            @as(f32, @floatFromInt(src_pixels[idx1].r)),
            @as(f32, @floatFromInt(src_pixels[idx2].r)),
            @as(f32, @floatFromInt(src_pixels[idx3].r)),
        };
        const g_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].g)),
            @as(f32, @floatFromInt(src_pixels[idx1].g)),
            @as(f32, @floatFromInt(src_pixels[idx2].g)),
            @as(f32, @floatFromInt(src_pixels[idx3].g)),
        };
        const b_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].b)),
            @as(f32, @floatFromInt(src_pixels[idx1].b)),
            @as(f32, @floatFromInt(src_pixels[idx2].b)),
            @as(f32, @floatFromInt(src_pixels[idx3].b)),
        };
        const a_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].a)),
            @as(f32, @floatFromInt(src_pixels[idx1].a)),
            @as(f32, @floatFromInt(src_pixels[idx2].a)),
            @as(f32, @floatFromInt(src_pixels[idx3].a)),
        };

        const weight = kernel[k];
        const weight_vec: Vec4f32 = @splat(weight);

        r_sum_vec += r_vals * weight_vec;
        g_sum_vec += g_vals * weight_vec;
        b_sum_vec += b_vals * weight_vec;
        a_sum_vec += a_vals * weight_vec;
    }

    // Store results
    const dst_idx0 = start_y * width + start_x;
    const dst_idx1 = start_y * width + start_x + 1;
    const dst_idx2 = start_y * width + start_x + 2;
    const dst_idx3 = start_y * width + start_x + 3;

    dst_pixels[dst_idx0].r = @as(u8, @intFromFloat(std.math.clamp(r_sum_vec[0], 0.0, 255.0)));
    dst_pixels[dst_idx0].g = @as(u8, @intFromFloat(std.math.clamp(g_sum_vec[0], 0.0, 255.0)));
    dst_pixels[dst_idx0].b = @as(u8, @intFromFloat(std.math.clamp(b_sum_vec[0], 0.0, 255.0)));
    dst_pixels[dst_idx0].a = @as(u8, @intFromFloat(std.math.clamp(a_sum_vec[0], 0.0, 255.0)));

    dst_pixels[dst_idx1].r = @as(u8, @intFromFloat(std.math.clamp(r_sum_vec[1], 0.0, 255.0)));
    dst_pixels[dst_idx1].g = @as(u8, @intFromFloat(std.math.clamp(g_sum_vec[1], 0.0, 255.0)));
    dst_pixels[dst_idx1].b = @as(u8, @intFromFloat(std.math.clamp(b_sum_vec[1], 0.0, 255.0)));
    dst_pixels[dst_idx1].a = @as(u8, @intFromFloat(std.math.clamp(a_sum_vec[1], 0.0, 255.0)));

    dst_pixels[dst_idx2].r = @as(u8, @intFromFloat(std.math.clamp(r_sum_vec[2], 0.0, 255.0)));
    dst_pixels[dst_idx2].g = @as(u8, @intFromFloat(std.math.clamp(g_sum_vec[2], 0.0, 255.0)));
    dst_pixels[dst_idx2].b = @as(u8, @intFromFloat(std.math.clamp(b_sum_vec[2], 0.0, 255.0)));
    dst_pixels[dst_idx2].a = @as(u8, @intFromFloat(std.math.clamp(a_sum_vec[2], 0.0, 255.0)));

    dst_pixels[dst_idx3].r = @as(u8, @intFromFloat(std.math.clamp(r_sum_vec[3], 0.0, 255.0)));
    dst_pixels[dst_idx3].g = @as(u8, @intFromFloat(std.math.clamp(g_sum_vec[3], 0.0, 255.0)));
    dst_pixels[dst_idx3].b = @as(u8, @intFromFloat(std.math.clamp(b_sum_vec[3], 0.0, 255.0)));
    dst_pixels[dst_idx3].a = @as(u8, @intFromFloat(std.math.clamp(a_sum_vec[3], 0.0, 255.0)));
}

/// Apply vertical 1D convolution to 4 pixels at once using SIMD
/// This processes 4 consecutive pixels vertically, handling boundary conditions
pub fn applyVerticalConvolutionSIMD4(
    src_pixels: []const img.color.Rgba32,
    dst_pixels: []img.color.Rgba32,
    width: usize,
    height: usize,
    kernel: []const f32,
    kernel_radius: usize,
    start_x: usize,
    start_y: usize,
) void {
    const kernel_size = kernel_radius * 2 + 1;

    // Process 4 pixels vertically if they fit
    if (start_y + 4 > height) return; // Not enough pixels for SIMD

    // Load 4 consecutive pixels and their neighbors
    var r_sum_vec: Vec4f32 = @splat(0.0);
    var g_sum_vec: Vec4f32 = @splat(0.0);
    var b_sum_vec: Vec4f32 = @splat(0.0);
    var a_sum_vec: Vec4f32 = @splat(0.0);

    // Apply kernel to each of the 4 pixels
    for (0..kernel_size) |k| {
        const offset = @as(i32, @intCast(k)) - @as(i32, @intCast(kernel_radius));

        // Calculate source positions for each of the 4 pixels
        const src_y0 = @as(i32, @intCast(start_y)) + offset;
        const src_y1 = @as(i32, @intCast(start_y + 1)) + offset;
        const src_y2 = @as(i32, @intCast(start_y + 2)) + offset;
        const src_y3 = @as(i32, @intCast(start_y + 3)) + offset;

        // Clamp to image boundaries
        const clamped_y0 = std.math.clamp(src_y0, 0, @as(i32, @intCast(height - 1)));
        const clamped_y1 = std.math.clamp(src_y1, 0, @as(i32, @intCast(height - 1)));
        const clamped_y2 = std.math.clamp(src_y2, 0, @as(i32, @intCast(height - 1)));
        const clamped_y3 = std.math.clamp(src_y3, 0, @as(i32, @intCast(height - 1)));

        // Load pixel values
        const idx0 = @as(usize, @intCast(clamped_y0)) * width + start_x;
        const idx1 = @as(usize, @intCast(clamped_y1)) * width + start_x;
        const idx2 = @as(usize, @intCast(clamped_y2)) * width + start_x;
        const idx3 = @as(usize, @intCast(clamped_y3)) * width + start_x;

        const r_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].r)),
            @as(f32, @floatFromInt(src_pixels[idx1].r)),
            @as(f32, @floatFromInt(src_pixels[idx2].r)),
            @as(f32, @floatFromInt(src_pixels[idx3].r)),
        };
        const g_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].g)),
            @as(f32, @floatFromInt(src_pixels[idx1].g)),
            @as(f32, @floatFromInt(src_pixels[idx2].g)),
            @as(f32, @floatFromInt(src_pixels[idx3].g)),
        };
        const b_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].b)),
            @as(f32, @floatFromInt(src_pixels[idx1].b)),
            @as(f32, @floatFromInt(src_pixels[idx2].b)),
            @as(f32, @floatFromInt(src_pixels[idx3].b)),
        };
        const a_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].a)),
            @as(f32, @floatFromInt(src_pixels[idx1].a)),
            @as(f32, @floatFromInt(src_pixels[idx2].a)),
            @as(f32, @floatFromInt(src_pixels[idx3].a)),
        };

        const weight = kernel[k];
        const weight_vec: Vec4f32 = @splat(weight);

        r_sum_vec += r_vals * weight_vec;
        g_sum_vec += g_vals * weight_vec;
        b_sum_vec += b_vals * weight_vec;
        a_sum_vec += a_vals * weight_vec;
    }

    // Store results
    const dst_idx0 = start_y * width + start_x;
    const dst_idx1 = (start_y + 1) * width + start_x;
    const dst_idx2 = (start_y + 2) * width + start_x;
    const dst_idx3 = (start_y + 3) * width + start_x;

    dst_pixels[dst_idx0].r = @as(u8, @intFromFloat(std.math.clamp(r_sum_vec[0], 0.0, 255.0)));
    dst_pixels[dst_idx0].g = @as(u8, @intFromFloat(std.math.clamp(g_sum_vec[0], 0.0, 255.0)));
    dst_pixels[dst_idx0].b = @as(u8, @intFromFloat(std.math.clamp(b_sum_vec[0], 0.0, 255.0)));
    dst_pixels[dst_idx0].a = @as(u8, @intFromFloat(std.math.clamp(a_sum_vec[0], 0.0, 255.0)));

    dst_pixels[dst_idx1].r = @as(u8, @intFromFloat(std.math.clamp(r_sum_vec[1], 0.0, 255.0)));
    dst_pixels[dst_idx1].g = @as(u8, @intFromFloat(std.math.clamp(g_sum_vec[1], 0.0, 255.0)));
    dst_pixels[dst_idx1].b = @as(u8, @intFromFloat(std.math.clamp(b_sum_vec[1], 0.0, 255.0)));
    dst_pixels[dst_idx1].a = @as(u8, @intFromFloat(std.math.clamp(a_sum_vec[1], 0.0, 255.0)));

    dst_pixels[dst_idx2].r = @as(u8, @intFromFloat(std.math.clamp(r_sum_vec[2], 0.0, 255.0)));
    dst_pixels[dst_idx2].g = @as(u8, @intFromFloat(std.math.clamp(g_sum_vec[2], 0.0, 255.0)));
    dst_pixels[dst_idx2].b = @as(u8, @intFromFloat(std.math.clamp(b_sum_vec[2], 0.0, 255.0)));
    dst_pixels[dst_idx2].a = @as(u8, @intFromFloat(std.math.clamp(a_sum_vec[2], 0.0, 255.0)));

    dst_pixels[dst_idx3].r = @as(u8, @intFromFloat(std.math.clamp(r_sum_vec[3], 0.0, 255.0)));
    dst_pixels[dst_idx3].g = @as(u8, @intFromFloat(std.math.clamp(g_sum_vec[3], 0.0, 255.0)));
    dst_pixels[dst_idx3].b = @as(u8, @intFromFloat(std.math.clamp(b_sum_vec[3], 0.0, 255.0)));
    dst_pixels[dst_idx3].a = @as(u8, @intFromFloat(std.math.clamp(a_sum_vec[3], 0.0, 255.0)));
}

/// Apply emboss kernel to 4 pixels at once using SIMD
/// This processes 4 consecutive pixels, each applying their own 3x3 emboss kernel
pub fn applyEmbossKernelSIMD4(
    src_pixels: []const img.color.Rgba32,
    dst_pixels: []img.color.Rgba32,
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
) void {
    // Emboss kernel weights: [-2, -1, 0; -1, 1, 1; 0, 1, 2]
    // For each of the 4 pixels, we need to apply this kernel

    // Process 4 pixels horizontally if they fit
    if (start_x + 4 > width) return; // Not enough pixels for SIMD

    var sum_r_vec: Vec4f32 = @splat(0.0);
    var sum_g_vec: Vec4f32 = @splat(0.0);
    var sum_b_vec: Vec4f32 = @splat(0.0);

    // Apply the 3x3 kernel to each of the 4 pixels
    // Kernel positions relative to each pixel:
    // (-1,-1): -2, (0,-1): -1, (1,-1): 0
    // (-1, 0): -1, (0, 0):  1, (1, 0): 1
    // (-1, 1):  0, (0, 1):  1, (1, 1): 2

    const kernel_weights = [_]i32{ -2, -1, 0, -1, 1, 1, 0, 1, 2 };
    const kernel_offsets = [_]struct { dx: i32, dy: i32 }{
        .{ .dx = -1, .dy = -1 }, .{ .dx = 0, .dy = -1 }, .{ .dx = 1, .dy = -1 },
        .{ .dx = -1, .dy = 0 },  .{ .dx = 0, .dy = 0 },  .{ .dx = 1, .dy = 0 },
        .{ .dx = -1, .dy = 1 },  .{ .dx = 0, .dy = 1 },  .{ .dx = 1, .dy = 1 },
    };

    // For each kernel position
    for (0..9) |k| {
        const weight = kernel_weights[k];
        const weight_vec: Vec4f32 = @splat(@as(f32, @floatFromInt(weight)));
        const offset = kernel_offsets[k];

        // Calculate source positions for each of the 4 pixels
        const base_x = @as(i32, @intCast(start_x));
        const src_x0 = base_x + offset.dx;
        const src_x1 = base_x + 1 + offset.dx;
        const src_x2 = base_x + 2 + offset.dx;
        const src_x3 = base_x + 3 + offset.dx;

        const base_y = @as(i32, @intCast(start_y));
        const src_y = base_y + offset.dy;

        // Clamp to image boundaries
        const clamped_x0 = std.math.clamp(src_x0, 0, @as(i32, @intCast(width - 1)));
        const clamped_x1 = std.math.clamp(src_x1, 0, @as(i32, @intCast(width - 1)));
        const clamped_x2 = std.math.clamp(src_x2, 0, @as(i32, @intCast(width - 1)));
        const clamped_x3 = std.math.clamp(src_x3, 0, @as(i32, @intCast(width - 1)));
        const clamped_y = std.math.clamp(src_y, 0, @as(i32, @intCast(height - 1)));

        // Load luminance values (since emboss works on grayscale)
        const idx0 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x0));
        const idx1 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x1));
        const idx2 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x2));
        const idx3 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x3));

        const lum0 = calculateLuminance(src_pixels[idx0]);
        const lum1 = calculateLuminance(src_pixels[idx1]);
        const lum2 = calculateLuminance(src_pixels[idx2]);
        const lum3 = calculateLuminance(src_pixels[idx3]);

        const lum_vec: Vec4f32 = [_]f32{ lum0, lum1, lum2, lum3 };

        sum_r_vec += lum_vec * weight_vec;
        sum_g_vec += lum_vec * weight_vec;
        sum_b_vec += lum_vec * weight_vec;
    }

    // Add bias and clamp to 0-255 to create the emboss effect
    const biased_r = sum_r_vec + @as(Vec4f32, @splat(128.0));
    const biased_g = sum_g_vec + @as(Vec4f32, @splat(128.0));
    const biased_b = sum_b_vec + @as(Vec4f32, @splat(128.0));

    // Clamp to 0-255 range
    const clamped_r = clampVec4F32(biased_r, ZERO_F32, MAX_U8_F32);
    const clamped_g = clampVec4F32(biased_g, ZERO_F32, MAX_U8_F32);
    const clamped_b = clampVec4F32(biased_b, ZERO_F32, MAX_U8_F32);

    // Convert to u8
    const final_r = clampVec4F32ToU8(clamped_r);
    const final_g = clampVec4F32ToU8(clamped_g);
    const final_b = clampVec4F32ToU8(clamped_b);

    // Store results
    const dst_idx0 = start_y * width + start_x;
    const dst_idx1 = start_y * width + start_x + 1;
    const dst_idx2 = start_y * width + start_x + 2;
    const dst_idx3 = start_y * width + start_x + 3;

    dst_pixels[dst_idx0].r = final_r[0];
    dst_pixels[dst_idx0].g = final_g[0];
    dst_pixels[dst_idx0].b = final_b[0];

    dst_pixels[dst_idx1].r = final_r[1];
    dst_pixels[dst_idx1].g = final_g[1];
    dst_pixels[dst_idx1].b = final_b[1];

    dst_pixels[dst_idx2].r = final_r[2];
    dst_pixels[dst_idx2].g = final_g[2];
    dst_pixels[dst_idx2].b = final_b[2];

    dst_pixels[dst_idx3].r = final_r[3];
    dst_pixels[dst_idx3].g = final_g[3];
    dst_pixels[dst_idx3].b = final_b[3];
}

/// Apply color emboss kernel to 4 pixels at once using SIMD
/// This processes 4 consecutive pixels, each applying their own 3x3 emboss kernel per channel
pub fn applyColorEmbossKernelSIMD4(
    src_pixels: []const img.color.Rgba32,
    dst_pixels: []img.color.Rgba32,
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    strength: f32,
) void {
    // Same emboss kernel as grayscale version, but applied per channel

    // Process 4 pixels horizontally if they fit
    if (start_x + 4 > width) return; // Not enough pixels for SIMD

    var sum_r_vec: Vec4f32 = @splat(0.0);
    var sum_g_vec: Vec4f32 = @splat(0.0);
    var sum_b_vec: Vec4f32 = @splat(0.0);

    // Apply the 3x3 kernel to each of the 4 pixels
    const kernel_weights = [_]i32{ -2, -1, 0, -1, 1, 1, 0, 1, 2 };
    const kernel_offsets = [_]struct { dx: i32, dy: i32 }{
        .{ .dx = -1, .dy = -1 }, .{ .dx = 0, .dy = -1 }, .{ .dx = 1, .dy = -1 },
        .{ .dx = -1, .dy = 0 },  .{ .dx = 0, .dy = 0 },  .{ .dx = 1, .dy = 0 },
        .{ .dx = -1, .dy = 1 },  .{ .dx = 0, .dy = 1 },  .{ .dx = 1, .dy = 1 },
    };

    // For each kernel position
    for (0..9) |k| {
        const weight = kernel_weights[k];
        const weight_vec: Vec4f32 = @splat(@as(f32, @floatFromInt(weight)));
        const offset = kernel_offsets[k];

        // Calculate source positions for each of the 4 pixels
        const base_x = @as(i32, @intCast(start_x));
        const src_x0 = base_x + offset.dx;
        const src_x1 = base_x + 1 + offset.dx;
        const src_x2 = base_x + 2 + offset.dx;
        const src_x3 = base_x + 3 + offset.dx;

        const base_y = @as(i32, @intCast(start_y));
        const src_y = base_y + offset.dy;

        // Clamp to image boundaries
        const clamped_x0 = std.math.clamp(src_x0, 0, @as(i32, @intCast(width - 1)));
        const clamped_x1 = std.math.clamp(src_x1, 0, @as(i32, @intCast(width - 1)));
        const clamped_x2 = std.math.clamp(src_x2, 0, @as(i32, @intCast(width - 1)));
        const clamped_x3 = std.math.clamp(src_x3, 0, @as(i32, @intCast(width - 1)));
        const clamped_y = std.math.clamp(src_y, 0, @as(i32, @intCast(height - 1)));

        // Load R channel values
        const idx0 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x0));
        const idx1 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x1));
        const idx2 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x2));
        const idx3 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x3));

        const r_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].r)),
            @as(f32, @floatFromInt(src_pixels[idx1].r)),
            @as(f32, @floatFromInt(src_pixels[idx2].r)),
            @as(f32, @floatFromInt(src_pixels[idx3].r)),
        };
        const g_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].g)),
            @as(f32, @floatFromInt(src_pixels[idx1].g)),
            @as(f32, @floatFromInt(src_pixels[idx2].g)),
            @as(f32, @floatFromInt(src_pixels[idx3].g)),
        };
        const b_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].b)),
            @as(f32, @floatFromInt(src_pixels[idx1].b)),
            @as(f32, @floatFromInt(src_pixels[idx2].b)),
            @as(f32, @floatFromInt(src_pixels[idx3].b)),
        };

        sum_r_vec += r_vals * weight_vec;
        sum_g_vec += g_vals * weight_vec;
        sum_b_vec += b_vals * weight_vec;
    }

    // Apply strength, add bias and clamp to 0-255 to create the emboss effect
    const strength_vec: Vec4f32 = @splat(strength);
    const biased_r = sum_r_vec * strength_vec + @as(Vec4f32, @splat(128.0));
    const biased_g = sum_g_vec * strength_vec + @as(Vec4f32, @splat(128.0));
    const biased_b = sum_b_vec * strength_vec + @as(Vec4f32, @splat(128.0));

    // Clamp to 0-255 range
    const clamped_r = clampVec4F32(biased_r, ZERO_F32, MAX_U8_F32);
    const clamped_g = clampVec4F32(biased_g, ZERO_F32, MAX_U8_F32);
    const clamped_b = clampVec4F32(biased_b, ZERO_F32, MAX_U8_F32);

    // Convert to u8
    const final_r = clampVec4F32ToU8(clamped_r);
    const final_g = clampVec4F32ToU8(clamped_g);
    const final_b = clampVec4F32ToU8(clamped_b);

    // Store results
    const dst_idx0 = start_y * width + start_x;
    const dst_idx1 = start_y * width + start_x + 1;
    const dst_idx2 = start_y * width + start_x + 2;
    const dst_idx3 = start_y * width + start_x + 3;

    dst_pixels[dst_idx0].r = final_r[0];
    dst_pixels[dst_idx0].g = final_g[0];
    dst_pixels[dst_idx0].b = final_b[0];
    dst_pixels[dst_idx0].a = src_pixels[dst_idx0].a;

    dst_pixels[dst_idx1].r = final_r[1];
    dst_pixels[dst_idx1].g = final_g[1];
    dst_pixels[dst_idx1].b = final_b[1];
    dst_pixels[dst_idx1].a = src_pixels[dst_idx1].a;

    dst_pixels[dst_idx2].r = final_r[2];
    dst_pixels[dst_idx2].g = final_g[2];
    dst_pixels[dst_idx2].b = final_b[2];
    dst_pixels[dst_idx2].a = src_pixels[dst_idx2].a;

    dst_pixels[dst_idx3].r = final_r[3];
    dst_pixels[dst_idx3].g = final_g[3];
    dst_pixels[dst_idx3].b = final_b[3];
    dst_pixels[dst_idx3].a = src_pixels[dst_idx3].a;
}

/// Apply Sobel edge detection kernel to 4 pixels at once using SIMD
/// This processes 4 consecutive pixels, each applying their own 3x3 Sobel kernels
pub fn applySobelKernelSIMD4(
    src_pixels: []const img.color.Rgba32,
    dst_pixels: []img.color.Rgba32,
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
) void {
    // Sobel kernels:
    // Gx: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    // Gy: [-1, -2, -1; 0, 0, 0; 1, 2, 1]

    // Process 4 pixels horizontally if they fit
    if (start_x + 4 > width) return; // Not enough pixels for SIMD

    var gx_r_vec: Vec4f32 = @splat(0.0);
    var gy_r_vec: Vec4f32 = @splat(0.0);
    var gx_g_vec: Vec4f32 = @splat(0.0);
    var gy_g_vec: Vec4f32 = @splat(0.0);
    var gx_b_vec: Vec4f32 = @splat(0.0);
    var gy_b_vec: Vec4f32 = @splat(0.0);

    // Gx kernel weights: [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    // Gy kernel weights: [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    const gx_weights = [_]i32{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    const gy_weights = [_]i32{ -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    const kernel_offsets = [_]struct { dx: i32, dy: i32 }{
        .{ .dx = -1, .dy = -1 }, .{ .dx = 0, .dy = -1 }, .{ .dx = 1, .dy = -1 },
        .{ .dx = -1, .dy = 0 },  .{ .dx = 0, .dy = 0 },  .{ .dx = 1, .dy = 0 },
        .{ .dx = -1, .dy = 1 },  .{ .dx = 0, .dy = 1 },  .{ .dx = 1, .dy = 1 },
    };

    // For each kernel position
    for (0..9) |k| {
        const gx_weight = gx_weights[k];
        const gy_weight = gy_weights[k];
        const gx_weight_vec: Vec4f32 = @splat(@as(f32, @floatFromInt(gx_weight)));
        const gy_weight_vec: Vec4f32 = @splat(@as(f32, @floatFromInt(gy_weight)));
        const offset = kernel_offsets[k];

        // Calculate source positions for each of the 4 pixels
        const base_x = @as(i32, @intCast(start_x));
        const src_x0 = base_x + offset.dx;
        const src_x1 = base_x + 1 + offset.dx;
        const src_x2 = base_x + 2 + offset.dx;
        const src_x3 = base_x + 3 + offset.dx;

        const base_y = @as(i32, @intCast(start_y));
        const src_y = base_y + offset.dy;

        // Clamp to image boundaries
        const clamped_x0 = std.math.clamp(src_x0, 0, @as(i32, @intCast(width - 1)));
        const clamped_x1 = std.math.clamp(src_x1, 0, @as(i32, @intCast(width - 1)));
        const clamped_x2 = std.math.clamp(src_x2, 0, @as(i32, @intCast(width - 1)));
        const clamped_x3 = std.math.clamp(src_x3, 0, @as(i32, @intCast(width - 1)));
        const clamped_y = std.math.clamp(src_y, 0, @as(i32, @intCast(height - 1)));

        // Load pixel values
        const idx0 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x0));
        const idx1 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x1));
        const idx2 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x2));
        const idx3 = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x3));

        const r_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].r)),
            @as(f32, @floatFromInt(src_pixels[idx1].r)),
            @as(f32, @floatFromInt(src_pixels[idx2].r)),
            @as(f32, @floatFromInt(src_pixels[idx3].r)),
        };
        const g_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].g)),
            @as(f32, @floatFromInt(src_pixels[idx1].g)),
            @as(f32, @floatFromInt(src_pixels[idx2].g)),
            @as(f32, @floatFromInt(src_pixels[idx3].g)),
        };
        const b_vals: Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(src_pixels[idx0].b)),
            @as(f32, @floatFromInt(src_pixels[idx1].b)),
            @as(f32, @floatFromInt(src_pixels[idx2].b)),
            @as(f32, @floatFromInt(src_pixels[idx3].b)),
        };

        gx_r_vec += r_vals * gx_weight_vec;
        gy_r_vec += r_vals * gy_weight_vec;
        gx_g_vec += g_vals * gx_weight_vec;
        gy_g_vec += g_vals * gy_weight_vec;
        gx_b_vec += b_vals * gx_weight_vec;
        gy_b_vec += b_vals * gy_weight_vec;
    }

    // Calculate gradient magnitude: sqrt(Gx^2 + Gy^2)
    const mag_r_vec = sqrtVec4F32(gx_r_vec * gx_r_vec + gy_r_vec * gy_r_vec);
    const mag_g_vec = sqrtVec4F32(gx_g_vec * gx_g_vec + gy_g_vec * gy_g_vec);
    const mag_b_vec = sqrtVec4F32(gx_b_vec * gx_b_vec + gy_b_vec * gy_b_vec);

    // Clamp to 0-255 range and convert to u8
    const clamped_r = clampVec4F32ToU8(mag_r_vec);
    const clamped_g = clampVec4F32ToU8(mag_g_vec);
    const clamped_b = clampVec4F32ToU8(mag_b_vec);

    // Store results
    const dst_idx0 = start_y * width + start_x;
    const dst_idx1 = start_y * width + start_x + 1;
    const dst_idx2 = start_y * width + start_x + 2;
    const dst_idx3 = start_y * width + start_x + 3;

    dst_pixels[dst_idx0].r = clamped_r[0];
    dst_pixels[dst_idx0].g = clamped_g[0];
    dst_pixels[dst_idx0].b = clamped_b[0];

    dst_pixels[dst_idx1].r = clamped_r[1];
    dst_pixels[dst_idx1].g = clamped_g[1];
    dst_pixels[dst_idx1].b = clamped_b[1];

    dst_pixels[dst_idx2].r = clamped_r[2];
    dst_pixels[dst_idx2].g = clamped_g[2];
    dst_pixels[dst_idx2].b = clamped_b[2];

    dst_pixels[dst_idx3].r = clamped_r[3];
    dst_pixels[dst_idx3].g = clamped_g[3];
    dst_pixels[dst_idx3].b = clamped_b[3];
}
