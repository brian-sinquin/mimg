const std = @import("std");
const img = @import("zigimg");

/// SIMD vector types for common operations
pub const Vec4f32 = @Vector(4, f32);
pub const Vec4u8 = @Vector(4, u8);
pub const Vec4i16 = @Vector(4, i16);

/// Common SIMD constants
pub const ZERO_F32: Vec4f32 = @splat(0.0);
pub const ONE_F32: Vec4f32 = @splat(1.0);
pub const MAX_U8_F32: Vec4f32 = @splat(255.0);
pub const MAX_U8_VEC: Vec4u8 = @splat(255);
pub const ZERO_U8_VEC: Vec4u8 = @splat(0);

/// Clamp a float value to u8 range (0-255)
pub fn clampF32ToU8(value: f32) u8 {
    return @as(u8, @intFromFloat(std.math.clamp(value, 0.0, 255.0)));
}

/// Clamp a vector of f32 values to u8 range
pub fn clampVec4F32ToU8(vec: Vec4f32) Vec4u8 {
    const clamped = @max(ZERO_F32, @min(vec, MAX_U8_F32));
    return @as(Vec4u8, @intFromFloat(clamped));
}

/// Convert RGBA32 pixel to f32 vector (RGB only, alpha preserved separately)
pub fn pixelToVec3F32(pixel: img.color.Rgba32) struct { Vec4f32, u8 } {
    return .{
        [_]f32{
            @as(f32, @floatFromInt(pixel.r)),
            @as(f32, @floatFromInt(pixel.g)),
            @as(f32, @floatFromInt(pixel.b)),
            0.0, // padding
        },
        pixel.a,
    };
}

/// Convert f32 vector back to RGBA32 pixel
pub fn vec3F32ToPixel(vec: Vec4f32, alpha: u8) img.color.Rgba32 {
    return img.color.Rgba32{
        .r = clampF32ToU8(vec[0]),
        .g = clampF32ToU8(vec[1]),
        .b = clampF32ToU8(vec[2]),
        .a = alpha,
    };
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

/// Process pixels using actual SIMD vector operations (4 pixels at a time)
pub fn processPixelsVectorSIMD4(
    pixels: []img.color.Rgba32,
    comptime processVecFn: fn (Vec4f32, Vec4f32, Vec4f32, Vec4u8) struct { Vec4f32, Vec4f32, Vec4f32 },
) void {
    var i: usize = 0;
    const pixel_count = pixels.len;

    // Process in chunks of 4 pixels using vector operations
    while (i + 4 <= pixel_count) : (i += 4) {
        // Load 4 pixels into vectors
        const r_vec: Vec4u8 = [_]u8{ pixels[i].r, pixels[i + 1].r, pixels[i + 2].r, pixels[i + 3].r };
        const g_vec: Vec4u8 = [_]u8{ pixels[i].g, pixels[i + 1].g, pixels[i + 2].g, pixels[i + 3].g };
        const b_vec: Vec4u8 = [_]u8{ pixels[i].b, pixels[i + 1].b, pixels[i + 2].b, pixels[i + 3].b };
        const a_vec: Vec4u8 = [_]u8{ pixels[i].a, pixels[i + 1].a, pixels[i + 2].a, pixels[i + 3].a };

        // Convert to f32 for processing
        const r_f32: Vec4f32 = @as(Vec4f32, @floatFromInt(r_vec));
        const g_f32: Vec4f32 = @as(Vec4f32, @floatFromInt(g_vec));
        const b_f32: Vec4f32 = @as(Vec4f32, @floatFromInt(b_vec));

        // Apply processing function
        const result = processVecFn(r_f32, g_f32, b_f32, a_vec);
        const new_r_vec = clampVec4F32ToU8(result[0]);
        const new_g_vec = clampVec4F32ToU8(result[1]);
        const new_b_vec = clampVec4F32ToU8(result[2]);

        // Store back
        pixels[i] = img.color.Rgba32{ .r = new_r_vec[0], .g = new_r_vec[1], .b = new_r_vec[2], .a = new_r_vec[3] };
        pixels[i + 1] = img.color.Rgba32{ .r = new_g_vec[0], .g = new_g_vec[1], .b = new_g_vec[2], .a = new_g_vec[3] };
        pixels[i + 2] = img.color.Rgba32{ .r = new_b_vec[0], .g = new_b_vec[1], .b = new_b_vec[2], .a = new_b_vec[3] };
        pixels[i + 3] = img.color.Rgba32{ .r = a_vec[0], .g = a_vec[1], .b = a_vec[2], .a = a_vec[3] };
    }

    // Handle remaining pixels with scalar operations
    while (i < pixel_count) : (i += 1) {
        // For remaining pixels, fall back to scalar processing
        // This would need to be implemented by the caller for remaining pixels
        // since we can't do vector operations on partial chunks
    }
}

/// Calculate luminance for a single pixel
pub fn calculateLuminance(pixel: img.color.Rgba32) f32 {
    const rf = @as(f32, @floatFromInt(pixel.r));
    const gf = @as(f32, @floatFromInt(pixel.g));
    const bf = @as(f32, @floatFromInt(pixel.b));
    return 0.299 * rf + 0.587 * gf + 0.114 * bf;
}

/// Calculate luminance for 4 pixels at once using SIMD
pub fn calculateLuminanceSIMD4(pixels: []const img.color.Rgba32, start_idx: usize) Vec4f32 {
    const r0 = @as(f32, @floatFromInt(pixels[start_idx].r));
    const g0 = @as(f32, @floatFromInt(pixels[start_idx].g));
    const b0 = @as(f32, @floatFromInt(pixels[start_idx].b));

    const r1 = @as(f32, @floatFromInt(pixels[start_idx + 1].r));
    const g1 = @as(f32, @floatFromInt(pixels[start_idx + 1].g));
    const b1 = @as(f32, @floatFromInt(pixels[start_idx + 1].b));

    const r2 = @as(f32, @floatFromInt(pixels[start_idx + 2].r));
    const g2 = @as(f32, @floatFromInt(pixels[start_idx + 2].g));
    const b2 = @as(f32, @floatFromInt(pixels[start_idx + 2].b));

    const r3 = @as(f32, @floatFromInt(pixels[start_idx + 3].r));
    const g3 = @as(f32, @floatFromInt(pixels[start_idx + 3].g));
    const b3 = @as(f32, @floatFromInt(pixels[start_idx + 3].b));

    const r_vec: Vec4f32 = [_]f32{ r0, r1, r2, r3 };
    const g_vec: Vec4f32 = [_]f32{ g0, g1, g2, g3 };
    const b_vec: Vec4f32 = [_]f32{ b0, b1, b2, b3 };

    const r_weight: Vec4f32 = @splat(0.299);
    const g_weight: Vec4f32 = @splat(0.587);
    const b_weight: Vec4f32 = @splat(0.114);

    return r_vec * r_weight + g_vec * g_weight + b_vec * b_weight;
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
