const img = @import("zigimg");
const std = @import("std");
const Context = @import("../core/types.zig").Context;
const utils = @import("../core/utils.zig");
const math = std.math;
const simd = @import("../utils/simd_utils.zig");

pub fn blurImage(ctx: *Context, args: anytype) !void {
    const kernel_size = args[0];

    // Input validation with descriptive error messages
    if (kernel_size <= 0) {
        std.log.err("Blur kernel size must be positive, got {d}", .{kernel_size});
        return error.KernelSizeError;
    }
    if (kernel_size % 2 == 0) {
        std.log.err("Blur kernel size must be odd, got {d}. Use {d} instead?", .{ kernel_size, kernel_size + 1 });
        return error.KernelSizeError;
    }
    if (kernel_size > 15) {
        std.log.warn("Large blur kernel size {d} may be slow", .{kernel_size});
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying blur with kernel size {}", .{kernel_size});
    utils.logMemoryUsage(ctx, "Blur start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.copyToTempBuffer(pixels);

    // Generate separable box filter kernel (uniform weights)
    const kernel_radius = kernel_size / 2;
    const kernel = try ctx.allocator.alloc(f32, kernel_size);
    defer ctx.allocator.free(kernel);

    // Box filter: all weights are equal (1/kernel_size)
    const weight = 1.0 / @as(f32, @floatFromInt(kernel_size));
    @memset(kernel, weight);

    // Apply separable blur: horizontal pass
    applyHorizontalConvolutionSIMD(pixels, temp_pixels, width, height, kernel, kernel_radius);
    ctx.reportProgress(1, 2, "blur");

    // Apply separable blur: vertical pass
    applyVerticalConvolutionSIMD(temp_pixels, pixels, width, height, kernel, kernel_radius);
    ctx.reportProgress(2, 2, "blur");

    utils.logMemoryUsage(ctx, "Blur end");
}

/// Generate a 1D Gaussian kernel for separable blur
fn generateGaussianKernel1D(allocator: std.mem.Allocator, sigma: f64, kernel_radius: usize) ![]f32 {
    const kernel_size = kernel_radius * 2 + 1;
    const kernel = try allocator.alloc(f32, kernel_size);

    var sum: f32 = 0.0;
    const two_sigma_sq = 2.0 * sigma * sigma;

    for (0..kernel_size) |i| {
        const x = @as(f32, @floatFromInt(i)) - @as(f32, @floatFromInt(kernel_radius));
        const weight = @as(f32, @floatCast(std.math.exp(@as(f64, -(x * x)) / two_sigma_sq)));
        kernel[i] = weight;
        sum += weight;
    }

    // Normalize kernel
    for (kernel) |*w| {
        w.* /= sum;
    }

    return kernel;
}

/// SIMD-optimized horizontal convolution using 4-pixel vector operations
fn applyHorizontalConvolutionSIMD(
    src_pixels: []const img.color.Rgba32,
    dst_pixels: []img.color.Rgba32,
    width: usize,
    height: usize,
    kernel: []const f32,
    kernel_radius: usize,
) void {
    // Process rows
    for (0..height) |y| {
        var x: usize = 0;

        // Process 4 pixels at a time using SIMD where possible
        while (x + 4 <= width) : (x += 4) {
            simd.applyHorizontalConvolutionSIMD4(src_pixels, dst_pixels, width, kernel, kernel_radius, y, x);
        }

        // Handle remaining pixels with scalar operations
        while (x < width) : (x += 1) {
            var r_sum: f32 = 0.0;
            var g_sum: f32 = 0.0;
            var b_sum: f32 = 0.0;
            var a_sum: f32 = 0.0;

            const kernel_size = kernel_radius * 2 + 1;
            for (0..kernel_size) |k| {
                const offset = @as(i32, @intCast(k)) - @as(i32, @intCast(kernel_radius));
                const px = @as(i32, @intCast(x)) + offset;

                if (px >= 0 and px < @as(i32, @intCast(width))) {
                    const idx = y * width + @as(usize, @intCast(px));
                    const weight = kernel[k];
                    r_sum += @as(f32, @floatFromInt(src_pixels[idx].r)) * weight;
                    g_sum += @as(f32, @floatFromInt(src_pixels[idx].g)) * weight;
                    b_sum += @as(f32, @floatFromInt(src_pixels[idx].b)) * weight;
                    a_sum += @as(f32, @floatFromInt(src_pixels[idx].a)) * weight;
                }
            }

            const idx = y * width + x;
            dst_pixels[idx].r = @as(u8, @intFromFloat(std.math.clamp(r_sum, 0.0, 255.0)));
            dst_pixels[idx].g = @as(u8, @intFromFloat(std.math.clamp(g_sum, 0.0, 255.0)));
            dst_pixels[idx].b = @as(u8, @intFromFloat(std.math.clamp(b_sum, 0.0, 255.0)));
            dst_pixels[idx].a = @as(u8, @intFromFloat(std.math.clamp(a_sum, 0.0, 255.0)));
        }
    }
}

/// SIMD-optimized vertical convolution using 4-pixel vector operations
fn applyVerticalConvolutionSIMD(
    src_pixels: []const img.color.Rgba32,
    dst_pixels: []img.color.Rgba32,
    width: usize,
    height: usize,
    kernel: []const f32,
    kernel_radius: usize,
) void {
    // Process columns
    for (0..width) |x| {
        var y: usize = 0;

        // Process 4 pixels at a time using SIMD where possible
        while (y + 4 <= height) : (y += 4) {
            simd.applyVerticalConvolutionSIMD4(src_pixels, dst_pixels, width, height, kernel, kernel_radius, x, y);
        }

        // Handle remaining pixels with scalar operations
        while (y < height) : (y += 1) {
            var r_sum: f32 = 0.0;
            var g_sum: f32 = 0.0;
            var b_sum: f32 = 0.0;
            var a_sum: f32 = 0.0;

            const kernel_size = kernel_radius * 2 + 1;
            for (0..kernel_size) |k| {
                const offset = @as(i32, @intCast(k)) - @as(i32, @intCast(kernel_radius));
                const py = @as(i32, @intCast(y)) + offset;

                if (py >= 0 and py < @as(i32, @intCast(height))) {
                    const idx = @as(usize, @intCast(py)) * width + x;
                    const weight = kernel[k];
                    r_sum += @as(f32, @floatFromInt(src_pixels[idx].r)) * weight;
                    g_sum += @as(f32, @floatFromInt(src_pixels[idx].g)) * weight;
                    b_sum += @as(f32, @floatFromInt(src_pixels[idx].b)) * weight;
                    a_sum += @as(f32, @floatFromInt(src_pixels[idx].a)) * weight;
                }
            }

            const idx = y * width + x;
            dst_pixels[idx].r = @as(u8, @intFromFloat(std.math.clamp(r_sum, 0.0, 255.0)));
            dst_pixels[idx].g = @as(u8, @intFromFloat(std.math.clamp(g_sum, 0.0, 255.0)));
            dst_pixels[idx].b = @as(u8, @intFromFloat(std.math.clamp(b_sum, 0.0, 255.0)));
            dst_pixels[idx].a = @as(u8, @intFromFloat(std.math.clamp(a_sum, 0.0, 255.0)));
        }
    }
}

pub fn gaussianBlurImage(ctx: *Context, args: anytype) !void {
    const sigma = args[0];

    // Input validation with descriptive error messages
    if (sigma <= 0) {
        std.log.err("Gaussian blur sigma must be positive, got {d}", .{sigma});
        return error.InvalidSigma;
    }
    if (sigma > 5.0) {
        std.log.warn("Large Gaussian blur sigma {d} may be slow and produce poor results", .{sigma});
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying Gaussian blur with sigma {d:.2}", .{sigma});
    utils.logMemoryUsage(ctx, "Gaussian blur start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Calculate kernel size (should be odd and cover ~3*sigma)
    const kernel_radius = @as(usize, @intFromFloat(@ceil(sigma * 3.0)));

    // Generate 1D Gaussian kernel
    const kernel = try generateGaussianKernel1D(ctx.allocator, sigma, kernel_radius);
    defer ctx.allocator.free(kernel);

    // For separable blur, we need one temporary buffer
    const temp_pixels = try ctx.copyToTempBuffer(pixels);

    // Apply separable blur: horizontal pass
    applyHorizontalConvolutionSIMD(pixels, temp_pixels, width, height, kernel, kernel_radius);
    ctx.reportProgress(1, 2, "gaussian-blur");

    // Apply separable blur: vertical pass
    applyVerticalConvolutionSIMD(temp_pixels, pixels, width, height, kernel, kernel_radius);
    ctx.reportProgress(2, 2, "gaussian-blur");

    utils.logMemoryUsage(ctx, "Gaussian blur end");
}

pub fn sharpenImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying sharpen effect", .{});
    utils.logMemoryUsage(ctx, "Sharpen start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.copyToTempBuffer(pixels);

    // Sharpening kernel (3x3):
    // [0, -1, 0]
    // [-1, 5, -1]
    // [0, -1, 0]
    const sharpen_kernel = [_]i32{
        0,  -1, 0,
        -1, 5,  -1,
        0,  -1, 0,
    };

    // Apply 2D convolution with the sharpen kernel
    for (0..height) |y| {
        for (0..width) |x| {
            var r_sum: i32 = 0;
            var g_sum: i32 = 0;
            var b_sum: i32 = 0;

            // Apply 3x3 kernel
            for (0..3) |ky| {
                for (0..3) |kx| {
                    const offset_x = @as(i32, @intCast(kx)) - 1;
                    const offset_y = @as(i32, @intCast(ky)) - 1;
                    const nx_i32 = @as(i32, @intCast(x)) + offset_x;
                    const ny_i32 = @as(i32, @intCast(y)) + offset_y;

                    // Use edge replication for borders
                    const clamped_x = std.math.clamp(nx_i32, 0, @as(i32, @intCast(width - 1)));
                    const clamped_y = std.math.clamp(ny_i32, 0, @as(i32, @intCast(height - 1)));
                    const idx = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x));

                    const kernel_idx = ky * 3 + kx;
                    const weight = sharpen_kernel[kernel_idx];

                    r_sum += @as(i32, @intCast(temp_pixels[idx].r)) * weight;
                    g_sum += @as(i32, @intCast(temp_pixels[idx].g)) * weight;
                    b_sum += @as(i32, @intCast(temp_pixels[idx].b)) * weight;
                }
            }

            // Clamp results to valid range using inline helper
            const idx = y * width + x;
            pixels[idx].r = utils.clampU8(@as(f32, @floatFromInt(r_sum)));
            pixels[idx].g = utils.clampU8(@as(f32, @floatFromInt(g_sum)));
            pixels[idx].b = utils.clampU8(@as(f32, @floatFromInt(b_sum)));
        }
    }

    utils.logMemoryUsage(ctx, "Sharpen end");
}

pub fn embossImage(ctx: *Context, args: anytype) !void {
    const strength: f32 = args[0];
    if (strength <= 0.0 or strength > 5.0) {
        std.log.err("Emboss strength must be in range (0.0, 5.0], got {d}", .{strength});
        return error.InvalidStrength;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying emboss effect with strength {d:.2}", .{strength});
    utils.logMemoryUsage(ctx, "Emboss start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create grayscale version for processing
    const gray_pixels = try ctx.allocator.alloc(u8, width * height);
    defer ctx.allocator.free(gray_pixels);

    // Use utility function for luminance calculation - more compact and clear
    for (pixels, 0..) |pixel, i| {
        gray_pixels[i] = utils.rgbToLuminanceU8(pixel.r, pixel.g, pixel.b);
    }

    // Emboss kernel: creates 3D-like raised effect
    // [-2, -1,  0]
    // [-1,  1,  1]
    // [ 0,  1,  2]
    const kernel = [_][3]f32{
        .{ -2.0, -1.0, 0.0 },
        .{ -1.0, 1.0, 1.0 },
        .{ 0.0, 1.0, 2.0 },
    };

    // Apply emboss convolution
    for (0..height) |y| {
        for (0..width) |x| {
            var sum: f32 = 0.0;

            // Apply 3x3 kernel with edge replication
            for (0..3) |ky| {
                for (0..3) |kx| {
                    const offset_x = @as(i32, @intCast(kx)) - 1;
                    const offset_y = @as(i32, @intCast(ky)) - 1;
                    const nx = @as(i32, @intCast(x)) + offset_x;
                    const ny = @as(i32, @intCast(y)) + offset_y;

                    // Clamp to image boundaries (edge replication)
                    const clamped_x = std.math.clamp(nx, 0, @as(i32, @intCast(width - 1)));
                    const clamped_y = std.math.clamp(ny, 0, @as(i32, @intCast(height - 1)));
                    const idx = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x));

                    sum += @as(f32, @floatFromInt(gray_pixels[idx])) * kernel[ky][kx] * strength;
                }
            }

            // Add 128 bias then clamp using inline helper
            const emboss_value = utils.clampU8(sum + 128.0);

            const idx = y * width + x;
            pixels[idx].r = emboss_value;
            pixels[idx].g = emboss_value;
            pixels[idx].b = emboss_value;
        }
    }

    utils.logMemoryUsage(ctx, "Emboss end");
}

pub fn vignetteImage(ctx: *Context, args: anytype) !void {
    const intensity = args[0];
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying vignette effect with intensity {d:.2}", .{intensity});

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    const center_x = @as(f32, @floatFromInt(width)) * utils.INV_2;
    const center_y = @as(f32, @floatFromInt(height)) * utils.INV_2;
    const max_distance = @sqrt(center_x * center_x + center_y * center_y);
    const inv_max_distance = 1.0 / max_distance;

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

        // Apply vignette effect using SIMD
        const result = simd.applyVignetteSIMD4(r_vec, g_vec, b_vec, x_vec, y_vec, center_x, center_y, inv_max_distance, intensity);
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
        const y = i / width;
        const x = i % width;

        const dy = @as(f32, @floatFromInt(y)) - center_y;
        const dx = @as(f32, @floatFromInt(x)) - center_x;
        const distance = @sqrt(dx * dx + dy * dy);
        const normalized_distance = distance * inv_max_distance;

        // Vignette factor: closer to center = 1.0, edges = intensity
        const vignette_factor = 1.0 - (normalized_distance * intensity);

        pixels[i].r = utils.clampU8(@as(f32, @floatFromInt(pixels[i].r)) * vignette_factor);
        pixels[i].g = utils.clampU8(@as(f32, @floatFromInt(pixels[i].g)) * vignette_factor);
        pixels[i].b = utils.clampU8(@as(f32, @floatFromInt(pixels[i].b)) * vignette_factor);
    }
}

pub fn edgeDetectImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying edge detection (Sobel operator)", .{});
    utils.logMemoryUsage(ctx, "Edge detect start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.copyToTempBuffer(pixels);

    // Process pixels in SIMD chunks of 4
    var y: usize = 0;
    while (y < height) : (y += 1) {
        var x: usize = 0;
        while (x + 4 <= width) : (x += 4) {
            // Apply SIMD Sobel kernel to 4 pixels at once
            simd.applySobelKernelSIMD4(temp_pixels, pixels, width, height, x, y);
        }

        // Handle remaining pixels in this row with scalar processing
        while (x < width) : (x += 1) {
            // Calculate Sobel gradients for each channel
            const gx_r = applySobelGxChannel(temp_pixels, width, height, x, y, .r);
            const gy_r = applySobelGyChannel(temp_pixels, width, height, x, y, .r);
            const gx_g = applySobelGxChannel(temp_pixels, width, height, x, y, .g);
            const gy_g = applySobelGyChannel(temp_pixels, width, height, x, y, .g);
            const gx_b = applySobelGxChannel(temp_pixels, width, height, x, y, .b);
            const gy_b = applySobelGyChannel(temp_pixels, width, height, x, y, .b);

            // Calculate magnitude
            const magnitude_r = @sqrt(@as(f32, @floatFromInt(gx_r * gx_r + gy_r * gy_r)));
            const magnitude_g = @sqrt(@as(f32, @floatFromInt(gx_g * gx_g + gy_g * gy_g)));
            const magnitude_b = @sqrt(@as(f32, @floatFromInt(gx_b * gx_b + gy_b * gy_b)));

            const idx = y * width + x;
            pixels[idx].r = utils.clampU8(magnitude_r);
            pixels[idx].g = utils.clampU8(magnitude_g);
            pixels[idx].b = utils.clampU8(magnitude_b);
        }
    }

    utils.logMemoryUsage(ctx, "Edge detect end");
}

pub fn medianFilterImage(ctx: *Context, args: anytype) !void {
    const kernel_size = args[0];

    // Input validation
    if (kernel_size <= 0 or kernel_size % 2 == 0) {
        return error.KernelSizeError;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying median filter with kernel size {d}", .{kernel_size});
    utils.logMemoryUsage(ctx, "Median filter start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.copyToTempBuffer(pixels);

    const radius = kernel_size / 2;
    const window_size = kernel_size * kernel_size;

    // For very large images, process in tiles to reduce memory usage
    const max_tile_pixels = 1024 * 1024; // 1M pixels per tile
    const tile_height = @min(height, max_tile_pixels / width);
    if (tile_height == 0) return error.ImageTooLarge;

    // Create arrays to hold pixel values for sorting (reuse for each tile)
    const r_values = try ctx.allocator.alloc(u8, window_size);
    defer ctx.allocator.free(r_values);
    const g_values = try ctx.allocator.alloc(u8, window_size);
    defer ctx.allocator.free(g_values);
    const b_values = try ctx.allocator.alloc(u8, window_size);
    defer ctx.allocator.free(b_values);

    // Process image in tiles
    var tile_start_y: usize = radius;
    var tile_count: usize = 0;
    const total_tiles = (height - 2 * radius + tile_height - 1) / tile_height; // Approximate

    while (tile_start_y < height - radius) {
        const tile_end_y = @min(tile_start_y + tile_height - radius * 2, height - radius);

        for (tile_start_y..tile_end_y) |y| {
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

        tile_start_y = tile_end_y;
        tile_count += 1;
        ctx.reportProgress(tile_count, total_tiles, "median-filter");
    }

    utils.logMemoryUsage(ctx, "Median filter end");
}

pub fn addNoiseImage(ctx: *Context, args: anytype) !void {
    const amount = args[0];

    // Input validation with descriptive error messages
    if (amount < 0.0 or amount > 1.0) {
        std.log.err("Noise amount must be in range 0.0-1.0, got {d}", .{amount});
        return error.InvalidNoiseAmount;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Adding noise with amount {d:.2}", .{amount});

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

pub fn pixelateImage(ctx: *Context, args: anytype) !void {
    const block_size = args[0];

    // Input validation
    if (block_size <= 0) {
        return error.InvalidBlockSize;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying pixelate effect with block size {d}", .{block_size});

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Use helper function to apply pixelation to entire image
    applyPixelateToRegion(pixels, width, 0, 0, width, height, block_size);
}

pub fn oilPaintingImage(ctx: *Context, args: anytype) !void {
    const radius = args[0];

    // Input validation
    if (radius <= 0) {
        return error.InvalidRadius;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying oil painting effect with radius {d}", .{radius});
    utils.logMemoryUsage(ctx, "Oil painting start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.copyToTempBuffer(pixels);

    // Oil painting uses intensity levels for a painted look
    const intensity_levels: usize = 20;

    // For large images, process in tiles
    const max_tile_pixels = 512 * 512; // Smaller tiles for oil painting
    const tile_height = @min(height, max_tile_pixels / width);
    if (tile_height == 0) return error.ImageTooLarge;

    // Allocate intensity tracking arrays (reused for each tile)
    const intensity_count = try ctx.allocator.alloc(u32, intensity_levels);
    defer ctx.allocator.free(intensity_count);
    const avg_r = try ctx.allocator.alloc(f32, intensity_levels);
    defer ctx.allocator.free(avg_r);
    const avg_g = try ctx.allocator.alloc(f32, intensity_levels);
    defer ctx.allocator.free(avg_g);
    const avg_b = try ctx.allocator.alloc(f32, intensity_levels);
    defer ctx.allocator.free(avg_b);

    // Process image in tiles
    var tile_start_y: usize = 0;
    var tile_count: usize = 0;
    const total_tiles = (height + tile_height - 1) / tile_height;

    while (tile_start_y < height) {
        const tile_end_y = @min(tile_start_y + tile_height, height);

        for (tile_start_y..tile_end_y) |y| {
            for (0..width) |x| {
                // Reset intensity tracking for this pixel
                @memset(intensity_count, 0);
                @memset(avg_r, 0);
                @memset(avg_g, 0);
                @memset(avg_b, 0);

                // Scan neighborhood
                const y_start = if (y >= radius) y - radius else 0;
                const y_end = @min(y + radius + 1, height);
                const x_start = if (x >= radius) x - radius else 0;
                const x_end = @min(x + radius + 1, width);

                for (y_start..y_end) |ny| {
                    for (x_start..x_end) |nx| {
                        const idx = ny * width + nx;
                        const rf = utils.u8ToF32(temp_pixels[idx].r);
                        const gf = utils.u8ToF32(temp_pixels[idx].g);
                        const bf = utils.u8ToF32(temp_pixels[idx].b);

                        // Calculate intensity level using optimized constants
                        const intensity = (rf + gf + bf) * utils.INV_3;
                        const level = @min(@as(usize, @intFromFloat(intensity * utils.INV_255 * @as(f32, @floatFromInt(intensity_levels - 1)))), intensity_levels - 1);

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
                    const inv_count = 1.0 / @as(f32, @floatFromInt(max_count));
                    pixels[idx].r = utils.clampU8(avg_r[max_level] * inv_count);
                    pixels[idx].g = utils.clampU8(avg_g[max_level] * inv_count);
                    pixels[idx].b = utils.clampU8(avg_b[max_level] * inv_count);
                }
            }
        }

        tile_start_y = tile_end_y;
        tile_count += 1;
        ctx.reportProgress(tile_count, total_tiles, "oil-painting");
    }

    utils.logMemoryUsage(ctx, "Oil painting end");
}

pub fn denoiseImage(ctx: *Context, args: anytype) !void {
    const strength = args[0];

    // Input validation
    if (strength < 1 or strength > 10) {
        std.log.err("Denoise strength must be in range 1-10, got {d}", .{strength});
        return error.InvalidStrength;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying denoising with strength {d}", .{strength});
    utils.logMemoryUsage(ctx, "Denoise start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.copyToTempBuffer(pixels);

    // Bilateral filter parameters
    const spatial_sigma = @as(f32, @floatFromInt(strength)) * 0.5; // Spatial weight
    const intensity_sigma = @as(f32, @floatFromInt(strength)) * 10.0; // Intensity weight
    const radius = @as(usize, @intFromFloat(@ceil(spatial_sigma * 2.0)));

    // Pre-compute Gaussian kernels for efficiency
    const kernel_size = radius * 2 + 1;
    const spatial_kernel = try ctx.allocator.alloc(f32, kernel_size * kernel_size);
    defer ctx.allocator.free(spatial_kernel);

    // Generate spatial Gaussian kernel
    const spatial_two_sigma_sq = 2.0 * spatial_sigma * spatial_sigma;
    const intensity_two_sigma_sq = 2.0 * intensity_sigma * intensity_sigma;

    for (0..kernel_size) |ky| {
        for (0..kernel_size) |kx| {
            const dx = @as(f32, @floatFromInt(kx)) - @as(f32, @floatFromInt(radius));
            const dy = @as(f32, @floatFromInt(ky)) - @as(f32, @floatFromInt(radius));
            const spatial_dist_sq = dx * dx + dy * dy;
            const spatial_weight = @exp(-spatial_dist_sq / spatial_two_sigma_sq);
            spatial_kernel[ky * kernel_size + kx] = spatial_weight;
        }
    }

    // Apply bilateral filter
    for (0..height) |y| {
        for (0..width) |x| {
            var r_sum: f32 = 0.0;
            var g_sum: f32 = 0.0;
            var b_sum: f32 = 0.0;
            var weight_sum: f32 = 0.0;

            // Get center pixel intensity
            const center_idx = y * width + x;
            const center_r = @as(f32, @floatFromInt(temp_pixels[center_idx].r));
            const center_g = @as(f32, @floatFromInt(temp_pixels[center_idx].g));
            const center_b = @as(f32, @floatFromInt(temp_pixels[center_idx].b));

            // Apply kernel
            for (0..kernel_size) |ky| {
                for (0..kernel_size) |kx| {
                    const nx_i32 = @as(i32, @intCast(x)) + @as(i32, @intCast(kx)) - @as(i32, @intCast(radius));
                    const ny_i32 = @as(i32, @intCast(y)) + @as(i32, @intCast(ky)) - @as(i32, @intCast(radius));

                    if (nx_i32 >= 0 and nx_i32 < @as(i32, @intCast(width)) and ny_i32 >= 0 and ny_i32 < @as(i32, @intCast(height))) {
                        const nx = @as(usize, @intCast(nx_i32));
                        const ny = @as(usize, @intCast(ny_i32));
                        const neighbor_idx = ny * width + nx;

                        const neighbor_r = @as(f32, @floatFromInt(temp_pixels[neighbor_idx].r));
                        const neighbor_g = @as(f32, @floatFromInt(temp_pixels[neighbor_idx].g));
                        const neighbor_b = @as(f32, @floatFromInt(temp_pixels[neighbor_idx].b));

                        // Calculate intensity difference
                        const dr = neighbor_r - center_r;
                        const dg = neighbor_g - center_g;
                        const db = neighbor_b - center_b;
                        const intensity_dist_sq = dr * dr + dg * dg + db * db;

                        // Bilateral weight = spatial weight * intensity weight
                        const spatial_weight = spatial_kernel[ky * kernel_size + kx];
                        const intensity_weight = @exp(-intensity_dist_sq / intensity_two_sigma_sq);
                        const total_weight = spatial_weight * intensity_weight;

                        r_sum += neighbor_r * total_weight;
                        g_sum += neighbor_g * total_weight;
                        b_sum += neighbor_b * total_weight;
                        weight_sum += total_weight;
                    }
                }
            }

            // Normalize by weight sum
            if (weight_sum > 0.0) {
                pixels[center_idx].r = @as(u8, @intFromFloat(std.math.clamp(r_sum / weight_sum, 0.0, 255.0)));
                pixels[center_idx].g = @as(u8, @intFromFloat(std.math.clamp(g_sum / weight_sum, 0.0, 255.0)));
                pixels[center_idx].b = @as(u8, @intFromFloat(std.math.clamp(b_sum / weight_sum, 0.0, 255.0)));
            }
        }

        // Report progress
        if (y % (height / 10) == 0) {
            ctx.reportProgress(y, height, "denoise");
        }
    }

    utils.logMemoryUsage(ctx, "Denoise end");
}

pub fn glowImage(ctx: *Context, args: anytype) !void {
    const intensity = args[0];
    const radius = args[1];

    // Input validation
    if (intensity < 0.0 or intensity > 1.0) {
        std.log.err("Glow intensity must be in range 0.0-1.0, got {d}", .{intensity});
        return error.InvalidIntensity;
    }
    if (radius < 1 or radius > 20) {
        std.log.err("Glow radius must be in range 1-20, got {d}", .{radius});
        return error.InvalidRadius;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying glow effect with intensity {d:.2} and radius {d}", .{ intensity, radius });
    utils.logMemoryUsage(ctx, "Glow start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create temporary buffers
    const original_pixels = try ctx.copyToTempBuffer(pixels);
    const glow_mask = try ctx.allocator.alloc(img.color.Rgba32, pixels.len);
    defer ctx.allocator.free(glow_mask);

    // Initialize glow mask to zero
    @memset(glow_mask, img.color.Rgba32{ .r = 0, .g = 0, .b = 0, .a = 255 });

    // Create glow mask from bright pixels
    const threshold = 200; // Brightness threshold for glow
    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            const pixel = original_pixels[idx];

            // Calculate luminance
            const luminance = @as(f32, @floatFromInt(pixel.r)) * 0.299 +
                @as(f32, @floatFromInt(pixel.g)) * 0.587 +
                @as(f32, @floatFromInt(pixel.b)) * 0.114;

            if (luminance > threshold) {
                // Add bright pixel to glow mask
                const glow_value = @as(u8, @intFromFloat(luminance * intensity));
                glow_mask[idx] = img.color.Rgba32{
                    .r = glow_value,
                    .g = glow_value,
                    .b = glow_value,
                    .a = 255,
                };
            }
        }
    }

    // Blur the glow mask using Gaussian blur
    const sigma = utils.u8ToF32(radius) * utils.INV_3;
    const kernel_radius = @as(usize, @intFromFloat(@ceil(sigma * 3.0)));

    // Generate 1D Gaussian kernel
    const kernel = try generateGaussianKernel1D(ctx.allocator, sigma, kernel_radius);
    defer ctx.allocator.free(kernel);

    // Create another temp buffer for the blurred glow mask
    const blurred_glow = try ctx.allocator.alloc(img.color.Rgba32, pixels.len);
    defer ctx.allocator.free(blurred_glow);

    // Copy glow mask to blurred_glow for horizontal pass
    @memcpy(blurred_glow, glow_mask);

    // Apply separable blur to glow mask: horizontal pass (using SIMD for better performance)
    applyHorizontalConvolutionSIMD(glow_mask, blurred_glow, width, height, kernel, kernel_radius);

    // Apply separable blur: vertical pass (using SIMD for better performance)
    applyVerticalConvolutionSIMD(blurred_glow, glow_mask, width, height, kernel, kernel_radius);

    // Add blurred glow back to original image
    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            const glow_pixel = glow_mask[idx];

            const new_r = utils.u8ToF32(original_pixels[idx].r) + utils.u8ToF32(glow_pixel.r);
            const new_g = utils.u8ToF32(original_pixels[idx].g) + utils.u8ToF32(glow_pixel.g);
            const new_b = utils.u8ToF32(original_pixels[idx].b) + utils.u8ToF32(glow_pixel.b);

            pixels[idx].r = utils.clampU8(new_r);
            pixels[idx].g = utils.clampU8(new_g);
            pixels[idx].b = utils.clampU8(new_b);
        }
    }

    utils.logMemoryUsage(ctx, "Glow end");
}

pub fn colorEmbossImage(ctx: *Context, args: anytype) !void {
    const strength: f32 = args[0];

    if (strength <= 0.0 or strength > 5.0) {
        std.log.err("Color emboss strength must be in range (0.0, 5.0], got {d}", .{strength});
        return error.InvalidStrength;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying color emboss effect with strength {d:.2}", .{strength});
    utils.logMemoryUsage(ctx, "Color emboss start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.copyToTempBuffer(pixels);

    // Process pixels in SIMD chunks of 4
    var y: usize = 0;
    while (y < height) : (y += 1) {
        var x: usize = 0;
        while (x + 4 <= width) : (x += 4) {
            // Apply SIMD color emboss kernel to 4 pixels at once
            simd.applyColorEmbossKernelSIMD4(temp_pixels, pixels, width, height, x, y, strength);
        }

        // Handle remaining pixels in this row with scalar processing
        while (x < width) : (x += 1) {
            // Calculate emboss values for each channel (apply strength)
            const r_emboss = @as(f32, @floatFromInt(applyEmbossKernelChannel(temp_pixels, width, height, x, y, .r))) * strength;
            const g_emboss = @as(f32, @floatFromInt(applyEmbossKernelChannel(temp_pixels, width, height, x, y, .g))) * strength;
            const b_emboss = @as(f32, @floatFromInt(applyEmbossKernelChannel(temp_pixels, width, height, x, y, .b))) * strength;

            // Add 128 bias and clamp to valid range
            const r_value = std.math.clamp(r_emboss + 128.0, 0.0, 255.0);
            const g_value = std.math.clamp(g_emboss + 128.0, 0.0, 255.0);
            const b_value = std.math.clamp(b_emboss + 128.0, 0.0, 255.0);

            const idx = y * width + x;
            pixels[idx].r = @as(u8, @intFromFloat(r_value));
            pixels[idx].g = @as(u8, @intFromFloat(g_value));
            pixels[idx].b = @as(u8, @intFromFloat(b_value));
        }
    }

    utils.logMemoryUsage(ctx, "Color emboss end");
}

const Channel = enum { r, g, b };

/// Apply emboss kernel to a specific color channel
fn applyEmbossKernelChannel(
    pixels: []const img.color.Rgba32,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    channel: Channel,
) i32 {
    var sum: i32 = 0;

    // Emboss kernel weights: [-2, -1, 0; -1, 1, 1; 0, 1, 2]
    // Top row
    sum += getChannelValue(pixels, width, height, x, y, -1, -1, channel) * (-2);
    sum += getChannelValue(pixels, width, height, x, y, 0, -1, channel) * (-1);
    sum += getChannelValue(pixels, width, height, x, y, 1, -1, channel) * 0;
    // Middle row
    sum += getChannelValue(pixels, width, height, x, y, -1, 0, channel) * (-1);
    sum += getChannelValue(pixels, width, height, x, y, 0, 0, channel) * 1;
    sum += getChannelValue(pixels, width, height, x, y, 1, 0, channel) * 1;
    // Bottom row
    sum += getChannelValue(pixels, width, height, x, y, -1, 1, channel) * 0;
    sum += getChannelValue(pixels, width, height, x, y, 0, 1, channel) * 1;
    sum += getChannelValue(pixels, width, height, x, y, 1, 1, channel) * 2;

    return sum;
}

/// Get channel value with edge replication for kernel operations
fn getChannelValue(
    pixels: []const img.color.Rgba32,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    offset_x: i32,
    offset_y: i32,
    channel: Channel,
) i32 {
    const nx_i32 = @as(i32, @intCast(x)) + offset_x;
    const ny_i32 = @as(i32, @intCast(y)) + offset_y;
    const clamped_x = std.math.clamp(nx_i32, 0, @as(i32, @intCast(width - 1)));
    const clamped_y = std.math.clamp(ny_i32, 0, @as(i32, @intCast(height - 1)));
    const idx = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x));

    return switch (channel) {
        .r => @as(i32, @intCast(pixels[idx].r)),
        .g => @as(i32, @intCast(pixels[idx].g)),
        .b => @as(i32, @intCast(pixels[idx].b)),
    };
}

pub fn tiltShiftImage(ctx: *Context, args: anytype) !void {
    const blur_strength = args[0];
    const focus_position = args[1]; // 0.0 = top, 1.0 = bottom
    const focus_width = args[2]; // 0.0 = no focus area, 1.0 = entire image in focus

    // Input validation
    if (blur_strength < 0.0 or blur_strength > 10.0) {
        std.log.err("Blur strength must be in range 0.0-10.0, got {d}", .{blur_strength});
        return error.InvalidBlurStrength;
    }
    if (focus_position < 0.0 or focus_position > 1.0) {
        std.log.err("Focus position must be in range 0.0-1.0, got {d}", .{focus_position});
        return error.InvalidFocusPosition;
    }
    if (focus_width < 0.0 or focus_width > 1.0) {
        std.log.err("Focus width must be in range 0.0-1.0, got {d}", .{focus_width});
        return error.InvalidFocusWidth;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying advanced tilt-shift effect - strength:{d:.1} pos:{d:.2} width:{d:.2}", .{ blur_strength, focus_position, focus_width });
    utils.logMemoryUsage(ctx, "Tilt-shift start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Early exit for zero blur
    if (blur_strength < 0.01) {
        utils.logVerbose(ctx, "Skipping tilt-shift due to minimal blur strength", .{});
        return;
    }

    // Create temporary buffers for two-pass blur
    const temp_pixels1 = try ctx.copyToTempBuffer(pixels);
    const temp_pixels2 = try ctx.allocator.alloc(img.color.Rgba32, pixels.len);
    defer ctx.allocator.free(temp_pixels2);

    // Calculate focus region with smoother falloff
    const focus_center = focus_position * @as(f32, @floatFromInt(height));
    const focus_half_width = (focus_width * @as(f32, @floatFromInt(height))) * utils.INV_2;
    const focus_start = focus_center - focus_half_width;
    const focus_end = focus_center + focus_half_width;

    // Enhanced falloff zones for more realistic depth of field
    const transition_zone = @max(focus_half_width * 0.3, 10.0); // Minimum 10px transition

    // Generate Gaussian kernel for maximum blur
    const max_sigma = blur_strength;
    const max_kernel_radius = @as(usize, @intFromFloat(@ceil(max_sigma * 3.0)));
    const max_kernel = try generateGaussianKernel1D(ctx.allocator, max_sigma, max_kernel_radius);
    defer ctx.allocator.free(max_kernel);

    // Calculate blur strength for each row using smooth distance function
    const blur_strengths = try ctx.allocator.alloc(f32, height);
    defer ctx.allocator.free(blur_strengths);

    for (0..height) |y| {
        const y_f = @as(f32, @floatFromInt(y));
        var distance_from_focus: f32 = 0.0;

        if (y_f < focus_start) {
            distance_from_focus = focus_start - y_f;
        } else if (y_f > focus_end) {
            distance_from_focus = y_f - focus_end;
        } else {
            // Inside focus zone - check distance to nearest edge for smooth transition
            const dist_to_start = y_f - focus_start;
            const dist_to_end = focus_end - y_f;
            const min_dist = @min(dist_to_start, dist_to_end);

            if (min_dist < transition_zone) {
                // Smooth transition within focus area
                distance_from_focus = @max(0.0, transition_zone - min_dist);
            }
        }

        // Apply smooth falloff curve (cubic for more natural look)
        const normalized_distance = @min(1.0, distance_from_focus / (@as(f32, @floatFromInt(height)) * 0.5));
        blur_strengths[y] = normalized_distance * normalized_distance * normalized_distance;
    }

    // First pass: Horizontal blur with variable strength
    for (0..height) |y| {
        const row_blur_factor = blur_strengths[y];

        if (row_blur_factor < 0.01) {
            // No blur needed, copy row directly
            const row_start = y * width;
            const row_end = row_start + width;
            @memcpy(temp_pixels1[row_start..row_end], pixels[row_start..row_end]);
            continue;
        }

        const effective_sigma = max_sigma * row_blur_factor;
        const effective_radius = @as(usize, @intFromFloat(@ceil(effective_sigma * 3.0)));

        // Apply horizontal convolution for this row
        for (0..width) |x| {
            var r_sum: f32 = 0.0;
            var g_sum: f32 = 0.0;
            var b_sum: f32 = 0.0;
            var weight_sum: f32 = 0.0;

            for (0..effective_radius * 2 + 1) |k| {
                const offset = @as(i32, @intCast(k)) - @as(i32, @intCast(effective_radius));
                const px = @as(i32, @intCast(x)) + offset;

                if (px >= 0 and px < @as(i32, @intCast(width))) {
                    // Calculate Gaussian weight on-the-fly for efficiency
                    const dist = @as(f32, @floatFromInt(@abs(offset)));
                    const weight = @exp(-(dist * dist) / (2.0 * effective_sigma * effective_sigma));

                    const idx = y * width + @as(usize, @intCast(px));
                    r_sum += @as(f32, @floatFromInt(pixels[idx].r)) * weight;
                    g_sum += @as(f32, @floatFromInt(pixels[idx].g)) * weight;
                    b_sum += @as(f32, @floatFromInt(pixels[idx].b)) * weight;
                    weight_sum += weight;
                }
            }

            const idx = y * width + x;
            if (weight_sum > 0.0) {
                temp_pixels1[idx].r = @as(u8, @intFromFloat(math.clamp(r_sum / weight_sum, 0.0, 255.0)));
                temp_pixels1[idx].g = @as(u8, @intFromFloat(math.clamp(g_sum / weight_sum, 0.0, 255.0)));
                temp_pixels1[idx].b = @as(u8, @intFromFloat(math.clamp(b_sum / weight_sum, 0.0, 255.0)));
                temp_pixels1[idx].a = pixels[idx].a;
            } else {
                temp_pixels1[idx] = pixels[idx];
            }
        }
    }

    // Second pass: Vertical blur with variable strength
    for (0..width) |x| {
        for (0..height) |y| {
            const row_blur_factor = blur_strengths[y];

            if (row_blur_factor < 0.01) {
                temp_pixels2[y * width + x] = temp_pixels1[y * width + x];
                continue;
            }

            const effective_sigma = max_sigma * row_blur_factor;
            const effective_radius = @as(usize, @intFromFloat(@ceil(effective_sigma * 3.0)));

            var r_sum: f32 = 0.0;
            var g_sum: f32 = 0.0;
            var b_sum: f32 = 0.0;
            var weight_sum: f32 = 0.0;

            for (0..effective_radius * 2 + 1) |k| {
                const offset = @as(i32, @intCast(k)) - @as(i32, @intCast(effective_radius));
                const py = @as(i32, @intCast(y)) + offset;

                if (py >= 0 and py < @as(i32, @intCast(height))) {
                    // Calculate Gaussian weight on-the-fly
                    const dist = @as(f32, @floatFromInt(@abs(offset)));
                    const weight = @exp(-(dist * dist) / (2.0 * effective_sigma * effective_sigma));

                    const idx = @as(usize, @intCast(py)) * width + x;
                    r_sum += @as(f32, @floatFromInt(temp_pixels1[idx].r)) * weight;
                    g_sum += @as(f32, @floatFromInt(temp_pixels1[idx].g)) * weight;
                    b_sum += @as(f32, @floatFromInt(temp_pixels1[idx].b)) * weight;
                    weight_sum += weight;
                }
            }

            const idx = y * width + x;
            if (weight_sum > 0.0) {
                temp_pixels2[idx].r = @as(u8, @intFromFloat(math.clamp(r_sum / weight_sum, 0.0, 255.0)));
                temp_pixels2[idx].g = @as(u8, @intFromFloat(math.clamp(g_sum / weight_sum, 0.0, 255.0)));
                temp_pixels2[idx].b = @as(u8, @intFromFloat(math.clamp(b_sum / weight_sum, 0.0, 255.0)));
                temp_pixels2[idx].a = temp_pixels1[idx].a;
            } else {
                temp_pixels2[idx] = temp_pixels1[idx];
            }
        }
    }

    // Apply subtle saturation boost to simulate miniature effect
    const saturation_boost = 1.1; // Slight boost for toy-like appearance
    for (0..pixels.len) |i| {
        const pixel = temp_pixels2[i];
        const r = utils.normalizeU8(pixel.r);
        const g = utils.normalizeU8(pixel.g);
        const b = utils.normalizeU8(pixel.b);

        // Convert to HSV, boost saturation, convert back
        const max_val = @max(@max(r, g), b);
        const min_val = @min(@min(r, g), b);
        const delta = max_val - min_val;

        if (delta > 0.001) { // Only boost if there's actual color
            const saturation = delta / max_val;
            const boosted_sat = @min(1.0, saturation * saturation_boost);
            const sat_factor = if (saturation > 0.001) boosted_sat / saturation else 1.0;

            const new_r = min_val + (r - min_val) * sat_factor;
            const new_g = min_val + (g - min_val) * sat_factor;
            const new_b = min_val + (b - min_val) * sat_factor;

            pixels[i].r = utils.clampU8(new_r * 255.0);
            pixels[i].g = utils.clampU8(new_g * 255.0);
            pixels[i].b = utils.clampU8(new_b * 255.0);
        } else {
            pixels[i].r = pixel.r;
            pixels[i].g = pixel.g;
            pixels[i].b = pixel.b;
        }
        pixels[i].a = pixel.a;
    }

    utils.logMemoryUsage(ctx, "Tilt-shift end");
}

pub fn edgeEnhancementImage(ctx: *Context, args: anytype) !void {
    const strength = args[0];

    // Input validation
    if (strength < 0.0 or strength > 2.0) {
        std.log.err("Edge enhancement strength must be in range 0.0-2.0, got {d}", .{strength});
        return error.InvalidStrength;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying edge enhancement with strength {d:.2}", .{strength});
    utils.logMemoryUsage(ctx, "Edge enhancement start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create temporary buffers
    const original_pixels = try ctx.copyToTempBuffer(pixels);
    const edge_pixels = try ctx.allocator.alloc(img.color.Rgba32, pixels.len);
    defer ctx.allocator.free(edge_pixels);

    // Apply edge detection to get edge mask using SIMD
    var y: usize = 0;
    while (y < height) : (y += 1) {
        var x: usize = 0;
        while (x + 4 <= width) : (x += 4) {
            // Apply SIMD Sobel kernel to 4 pixels at once
            simd.applySobelKernelSIMD4(original_pixels, edge_pixels, width, height, x, y);
        }

        // Handle remaining pixels in this row with scalar processing
        while (x < width) : (x += 1) {
            // Calculate Sobel gradients for each channel
            const gx_r = applySobelGxChannel(original_pixels, width, height, x, y, .r);
            const gy_r = applySobelGyChannel(original_pixels, width, height, x, y, .r);
            const gx_g = applySobelGxChannel(original_pixels, width, height, x, y, .g);
            const gy_g = applySobelGyChannel(original_pixels, width, height, x, y, .g);
            const gx_b = applySobelGxChannel(original_pixels, width, height, x, y, .b);
            const gy_b = applySobelGyChannel(original_pixels, width, height, x, y, .b);

            // Calculate gradient magnitude
            const mag_r = @sqrt(@as(f32, @floatFromInt(gx_r * gx_r + gy_r * gy_r)));
            const mag_g = @sqrt(@as(f32, @floatFromInt(gx_g * gx_g + gy_g * gy_g)));
            const mag_b = @sqrt(@as(f32, @floatFromInt(gx_b * gx_b + gy_b * gy_b)));

            const idx = y * width + x;
            edge_pixels[idx].r = @as(u8, @intFromFloat(std.math.clamp(mag_r, 0.0, 255.0)));
            edge_pixels[idx].g = @as(u8, @intFromFloat(std.math.clamp(mag_g, 0.0, 255.0)));
            edge_pixels[idx].b = @as(u8, @intFromFloat(std.math.clamp(mag_b, 0.0, 255.0)));
            edge_pixels[idx].a = 255;
        }
    }

    // Combine original image with enhanced edges using SIMD
    const strength_vec = @as(simd.Vec4f32, @splat(strength));
    const pixel_count = pixels.len;
    var i: usize = 0;

    // Process in chunks of 4 pixels using SIMD vector operations
    while (i + 4 <= pixel_count) : (i += 4) {
        // Load original pixels
        const orig_r_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(original_pixels[i].r)),
            @as(f32, @floatFromInt(original_pixels[i + 1].r)),
            @as(f32, @floatFromInt(original_pixels[i + 2].r)),
            @as(f32, @floatFromInt(original_pixels[i + 3].r)),
        };
        const orig_g_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(original_pixels[i].g)),
            @as(f32, @floatFromInt(original_pixels[i + 1].g)),
            @as(f32, @floatFromInt(original_pixels[i + 2].g)),
            @as(f32, @floatFromInt(original_pixels[i + 3].g)),
        };
        const orig_b_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(original_pixels[i].b)),
            @as(f32, @floatFromInt(original_pixels[i + 1].b)),
            @as(f32, @floatFromInt(original_pixels[i + 2].b)),
            @as(f32, @floatFromInt(original_pixels[i + 3].b)),
        };

        // Load edge pixels
        const edge_r_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(edge_pixels[i].r)),
            @as(f32, @floatFromInt(edge_pixels[i + 1].r)),
            @as(f32, @floatFromInt(edge_pixels[i + 2].r)),
            @as(f32, @floatFromInt(edge_pixels[i + 3].r)),
        };
        const edge_g_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(edge_pixels[i].g)),
            @as(f32, @floatFromInt(edge_pixels[i + 1].g)),
            @as(f32, @floatFromInt(edge_pixels[i + 2].g)),
            @as(f32, @floatFromInt(edge_pixels[i + 3].g)),
        };
        const edge_b_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(edge_pixels[i].b)),
            @as(f32, @floatFromInt(edge_pixels[i + 1].b)),
            @as(f32, @floatFromInt(edge_pixels[i + 2].b)),
            @as(f32, @floatFromInt(edge_pixels[i + 3].b)),
        };

        // Enhance edges: new_color = original + edge * strength
        const new_r_vec = orig_r_vec + edge_r_vec * strength_vec;
        const new_g_vec = orig_g_vec + edge_g_vec * strength_vec;
        const new_b_vec = orig_b_vec + edge_b_vec * strength_vec;

        // Clamp and convert back to u8
        const clamped_r = simd.clampVec4F32ToU8(new_r_vec);
        const clamped_g = simd.clampVec4F32ToU8(new_g_vec);
        const clamped_b = simd.clampVec4F32ToU8(new_b_vec);

        // Store back (preserve original alpha)
        pixels[i] = img.color.Rgba32{ .r = clamped_r[0], .g = clamped_g[0], .b = clamped_b[0], .a = original_pixels[i].a };
        pixels[i + 1] = img.color.Rgba32{ .r = clamped_r[1], .g = clamped_g[1], .b = clamped_b[1], .a = original_pixels[i + 1].a };
        pixels[i + 2] = img.color.Rgba32{ .r = clamped_r[2], .g = clamped_g[2], .b = clamped_b[2], .a = original_pixels[i + 2].a };
        pixels[i + 3] = img.color.Rgba32{ .r = clamped_r[3], .g = clamped_g[3], .b = clamped_b[3], .a = original_pixels[i + 3].a };
    }

    // Handle remaining pixels with scalar operations
    while (i < pixel_count) : (i += 1) {
        const original = original_pixels[i];
        const edge = edge_pixels[i];

        // Enhance edges by adding edge information to original
        const new_r = utils.u8ToF32(original.r) + utils.u8ToF32(edge.r) * strength;
        const new_g = utils.u8ToF32(original.g) + utils.u8ToF32(edge.g) * strength;
        const new_b = utils.u8ToF32(original.b) + utils.u8ToF32(edge.b) * strength;

        pixels[i].r = utils.clampU8(new_r);
        pixels[i].g = utils.clampU8(new_g);
        pixels[i].b = utils.clampU8(new_b);
        pixels[i].a = original.a;
    }

    utils.logMemoryUsage(ctx, "Edge enhancement end");
}

pub fn gradientLinearImage(ctx: *Context, args: anytype) !void {
    const start_color = args[0]; // Hex color string like "#FF0000"
    const end_color = args[1]; // Hex color string like "#0000FF"
    const angle = args[2]; // Angle in degrees (0 = horizontal, 90 = vertical)
    const opacity = args[3]; // 0.0 to 1.0

    // Input validation
    if (opacity < 0.0 or opacity > 1.0) {
        std.log.err("Gradient opacity must be in range 0.0-1.0, got {d}", .{opacity});
        return error.InvalidOpacity;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying linear gradient from {s} to {s} at {d}° with opacity {d:.2}", .{ start_color, end_color, angle, opacity });
    utils.logMemoryUsage(ctx, "Linear gradient start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Parse hex colors
    const start_rgba = try utils.parseHexColor(start_color);
    const end_rgba = try utils.parseHexColor(end_color);

    // Convert angle to radians
    const angle_rad = angle * utils.DEG_TO_RAD;

    // Calculate gradient direction vector
    const dir_x = @cos(angle_rad);
    const dir_y = @sin(angle_rad);

    // Pre-compute color vectors for SIMD
    const start_r_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(start_rgba.r))));
    const start_g_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(start_rgba.g))));
    const start_b_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(start_rgba.b))));
    const start_a_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(start_rgba.a))));

    const end_r_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(end_rgba.r))));
    const end_g_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(end_rgba.g))));
    const end_b_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(end_rgba.b))));
    const end_a_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(end_rgba.a))));

    const opacity_vec = @as(simd.Vec4f32, @splat(opacity));
    const one_minus_opacity_vec = @as(simd.Vec4f32, @splat(1.0 - opacity));

    const pixel_count = pixels.len;
    var i: usize = 0;

    // Process in chunks of 4 pixels using SIMD vector operations
    while (i + 4 <= pixel_count) : (i += 4) {
        // Calculate t values for 4 pixels
        const y0 = i / width;
        const x0 = i % width;
        const y1 = (i + 1) / width;
        const x1 = (i + 1) % width;
        const y2 = (i + 2) / width;
        const x2 = (i + 2) % width;
        const y3 = (i + 3) / width;
        const x3 = (i + 3) % width;

        const x_norm_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(x0)) / @as(f32, @floatFromInt(width - 1)) - 0.5,
            @as(f32, @floatFromInt(x1)) / @as(f32, @floatFromInt(width - 1)) - 0.5,
            @as(f32, @floatFromInt(x2)) / @as(f32, @floatFromInt(width - 1)) - 0.5,
            @as(f32, @floatFromInt(x3)) / @as(f32, @floatFromInt(width - 1)) - 0.5,
        };
        const y_norm_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(y0)) / @as(f32, @floatFromInt(height - 1)) - 0.5,
            @as(f32, @floatFromInt(y1)) / @as(f32, @floatFromInt(height - 1)) - 0.5,
            @as(f32, @floatFromInt(y2)) / @as(f32, @floatFromInt(height - 1)) - 0.5,
            @as(f32, @floatFromInt(y3)) / @as(f32, @floatFromInt(height - 1)) - 0.5,
        };

        const dir_x_vec = @as(simd.Vec4f32, @splat(dir_x));
        const dir_y_vec = @as(simd.Vec4f32, @splat(dir_y));

        // Project onto gradient direction: t = x_norm * dir_x + y_norm * dir_y + 0.5
        var t_vec = x_norm_vec * dir_x_vec + y_norm_vec * dir_y_vec;
        t_vec += @as(simd.Vec4f32, @splat(0.5));
        t_vec = simd.clampVec4F32(t_vec, @as(simd.Vec4f32, @splat(0.0)), @as(simd.Vec4f32, @splat(1.0)));

        // Interpolate colors: color = start * (1-t) + end * t
        const one_minus_t_vec = @as(simd.Vec4f32, @splat(1.0)) - t_vec;
        const grad_r_vec = start_r_vec * one_minus_t_vec + end_r_vec * t_vec;
        const grad_g_vec = start_g_vec * one_minus_t_vec + end_g_vec * t_vec;
        const grad_b_vec = start_b_vec * one_minus_t_vec + end_b_vec * t_vec;
        const grad_a_vec = start_a_vec * one_minus_t_vec + end_a_vec * t_vec;

        // Load original pixels
        const orig_r_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].r)),
            @as(f32, @floatFromInt(pixels[i + 1].r)),
            @as(f32, @floatFromInt(pixels[i + 2].r)),
            @as(f32, @floatFromInt(pixels[i + 3].r)),
        };
        const orig_g_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].g)),
            @as(f32, @floatFromInt(pixels[i + 1].g)),
            @as(f32, @floatFromInt(pixels[i + 2].g)),
            @as(f32, @floatFromInt(pixels[i + 3].g)),
        };
        const orig_b_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].b)),
            @as(f32, @floatFromInt(pixels[i + 1].b)),
            @as(f32, @floatFromInt(pixels[i + 2].b)),
            @as(f32, @floatFromInt(pixels[i + 3].b)),
        };
        const orig_a_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].a)),
            @as(f32, @floatFromInt(pixels[i + 1].a)),
            @as(f32, @floatFromInt(pixels[i + 2].a)),
            @as(f32, @floatFromInt(pixels[i + 3].a)),
        };

        // Blend: result = original * (1-opacity) + gradient * opacity
        const blended_r_vec = orig_r_vec * one_minus_opacity_vec + grad_r_vec * opacity_vec;
        const blended_g_vec = orig_g_vec * one_minus_opacity_vec + grad_g_vec * opacity_vec;
        const blended_b_vec = orig_b_vec * one_minus_opacity_vec + grad_b_vec * opacity_vec;
        const blended_a_vec = orig_a_vec * one_minus_opacity_vec + grad_a_vec * opacity_vec;

        // Clamp and convert back to u8
        const clamped_r = simd.clampVec4F32ToU8(blended_r_vec);
        const clamped_g = simd.clampVec4F32ToU8(blended_g_vec);
        const clamped_b = simd.clampVec4F32ToU8(blended_b_vec);
        const clamped_a = simd.clampVec4F32ToU8(blended_a_vec);

        // Store back
        pixels[i] = img.color.Rgba32{ .r = clamped_r[0], .g = clamped_g[0], .b = clamped_b[0], .a = clamped_a[0] };
        pixels[i + 1] = img.color.Rgba32{ .r = clamped_r[1], .g = clamped_g[1], .b = clamped_b[1], .a = clamped_a[1] };
        pixels[i + 2] = img.color.Rgba32{ .r = clamped_r[2], .g = clamped_g[2], .b = clamped_b[2], .a = clamped_a[2] };
        pixels[i + 3] = img.color.Rgba32{ .r = clamped_r[3], .g = clamped_g[3], .b = clamped_b[3], .a = clamped_a[3] };
    }

    // Handle remaining pixels with scalar operations
    // Pre-calculate normalization factors
    const inv_width_minus1 = 1.0 / @as(f32, @floatFromInt(width - 1));
    const inv_height_minus1 = 1.0 / @as(f32, @floatFromInt(height - 1));
    const one_minus_opacity = 1.0 - opacity;

    while (i < pixel_count) : (i += 1) {
        const y = i / width;
        const x = i % width;

        // Calculate position along gradient (from -1 to 1)
        const x_norm = @as(f32, @floatFromInt(x)) * inv_width_minus1 - 0.5;
        const y_norm = @as(f32, @floatFromInt(y)) * inv_height_minus1 - 0.5;

        // Project onto gradient direction
        const t = x_norm * dir_x + y_norm * dir_y + 0.5; // Add 0.5 to shift to 0-1 range
        const clamped_t = std.math.clamp(t, 0.0, 1.0);
        const one_minus_t = 1.0 - clamped_t;

        // Interpolate colors
        const r = @as(u8, @intFromFloat(utils.u8ToF32(start_rgba.r) * one_minus_t + utils.u8ToF32(end_rgba.r) * clamped_t));
        const g = @as(u8, @intFromFloat(utils.u8ToF32(start_rgba.g) * one_minus_t + utils.u8ToF32(end_rgba.g) * clamped_t));
        const b = @as(u8, @intFromFloat(utils.u8ToF32(start_rgba.b) * one_minus_t + utils.u8ToF32(end_rgba.b) * clamped_t));
        const a = @as(u8, @intFromFloat(utils.u8ToF32(start_rgba.a) * one_minus_t + utils.u8ToF32(end_rgba.a) * clamped_t));

        const original = pixels[i];

        // Blend with original image
        const blended_r = @as(u8, @intFromFloat(utils.u8ToF32(original.r) * one_minus_opacity + utils.u8ToF32(r) * opacity));
        const blended_g = @as(u8, @intFromFloat(utils.u8ToF32(original.g) * one_minus_opacity + utils.u8ToF32(g) * opacity));
        const blended_b = @as(u8, @intFromFloat(utils.u8ToF32(original.b) * one_minus_opacity + utils.u8ToF32(b) * opacity));
        const blended_a = @as(u8, @intFromFloat(utils.u8ToF32(original.a) * one_minus_opacity + utils.u8ToF32(a) * opacity));

        pixels[i].r = blended_r;
        pixels[i].g = blended_g;
        pixels[i].b = blended_b;
        pixels[i].a = blended_a;
    }

    utils.logMemoryUsage(ctx, "Linear gradient end");
}

pub fn gradientRadialImage(ctx: *Context, args: anytype) !void {
    const center_x = args[0]; // Center X position (0.0-1.0)
    const center_y = args[1]; // Center Y position (0.0-1.0)
    const start_color = args[2]; // Hex color string like "#FF0000"
    const end_color = args[3]; // Hex color string like "#0000FF"
    const radius = args[4]; // Radius (0.0-1.0, relative to image diagonal)
    const opacity = args[5]; // 0.0 to 1.0

    // Input validation
    if (center_x < 0.0 or center_x > 1.0 or center_y < 0.0 or center_y > 1.0) {
        std.log.err("Gradient center must be in range 0.0-1.0, got ({d}, {d})", .{ center_x, center_y });
        return error.InvalidCenter;
    }
    if (radius <= 0.0 or radius > 1.0) {
        std.log.err("Gradient radius must be in range 0.0-1.0, got {d}", .{radius});
        return error.InvalidRadius;
    }
    if (opacity < 0.0 or opacity > 1.0) {
        std.log.err("Gradient opacity must be in range 0.0-1.0, got {d}", .{opacity});
        return error.InvalidOpacity;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying radial gradient from {s} to {s} at ({d:.2}, {d:.2}) with radius {d:.2} and opacity {d:.2}", .{ start_color, end_color, center_x, center_y, radius, opacity });
    utils.logMemoryUsage(ctx, "Radial gradient start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Parse hex colors
    const start_rgba = try utils.parseHexColor(start_color);
    const end_rgba = try utils.parseHexColor(end_color);

    // Calculate center in pixel coordinates
    const cx = center_x * @as(f32, @floatFromInt(width));
    const cy = center_y * @as(f32, @floatFromInt(height));

    // Calculate maximum distance (from center to corner)
    const max_distance = @sqrt(cx * cx + cy * cy + (cx - @as(f32, @floatFromInt(width))) * (cx - @as(f32, @floatFromInt(width))) + (cy - @as(f32, @floatFromInt(height))) * (cy - @as(f32, @floatFromInt(height))));
    const effective_radius = radius * max_distance;

    // Pre-compute color vectors for SIMD
    const start_r_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(start_rgba.r))));
    const start_g_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(start_rgba.g))));
    const start_b_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(start_rgba.b))));
    const start_a_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(start_rgba.a))));

    const end_r_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(end_rgba.r))));
    const end_g_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(end_rgba.g))));
    const end_b_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(end_rgba.b))));
    const end_a_vec = @as(simd.Vec4f32, @splat(@as(f32, @floatFromInt(end_rgba.a))));

    const opacity_vec = @as(simd.Vec4f32, @splat(opacity));
    const one_minus_opacity_vec = @as(simd.Vec4f32, @splat(1.0 - opacity));

    const cx_vec = @as(simd.Vec4f32, @splat(cx));
    const cy_vec = @as(simd.Vec4f32, @splat(cy));
    const radius_vec = @as(simd.Vec4f32, @splat(effective_radius));

    const pixel_count = pixels.len;
    var i: usize = 0;

    // Process in chunks of 4 pixels using SIMD vector operations
    while (i + 4 <= pixel_count) : (i += 4) {
        // Calculate positions for 4 pixels
        const y0 = i / width;
        const x0 = i % width;
        const y1 = (i + 1) / width;
        const x1 = (i + 1) % width;
        const y2 = (i + 2) / width;
        const x2 = (i + 2) % width;
        const y3 = (i + 3) / width;
        const x3 = (i + 3) % width;

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

        // Calculate distance from center: distance = sqrt((x-cx)^2 + (y-cy)^2)
        const dx_vec = x_vec - cx_vec;
        const dy_vec = y_vec - cy_vec;
        const distance_vec = @sqrt(dx_vec * dx_vec + dy_vec * dy_vec);

        // Calculate interpolation factor: t = distance / radius (clamped to 0-1)
        const t_vec = simd.clampVec4F32(distance_vec / radius_vec, @as(simd.Vec4f32, @splat(0.0)), @as(simd.Vec4f32, @splat(1.0)));

        // Interpolate colors: color = start * (1-t) + end * t
        const one_minus_t_vec = @as(simd.Vec4f32, @splat(1.0)) - t_vec;
        const grad_r_vec = start_r_vec * one_minus_t_vec + end_r_vec * t_vec;
        const grad_g_vec = start_g_vec * one_minus_t_vec + end_g_vec * t_vec;
        const grad_b_vec = start_b_vec * one_minus_t_vec + end_b_vec * t_vec;
        const grad_a_vec = start_a_vec * one_minus_t_vec + end_a_vec * t_vec;

        // Load original pixels
        const orig_r_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].r)),
            @as(f32, @floatFromInt(pixels[i + 1].r)),
            @as(f32, @floatFromInt(pixels[i + 2].r)),
            @as(f32, @floatFromInt(pixels[i + 3].r)),
        };
        const orig_g_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].g)),
            @as(f32, @floatFromInt(pixels[i + 1].g)),
            @as(f32, @floatFromInt(pixels[i + 2].g)),
            @as(f32, @floatFromInt(pixels[i + 3].g)),
        };
        const orig_b_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].b)),
            @as(f32, @floatFromInt(pixels[i + 1].b)),
            @as(f32, @floatFromInt(pixels[i + 2].b)),
            @as(f32, @floatFromInt(pixels[i + 3].b)),
        };
        const orig_a_vec: simd.Vec4f32 = [_]f32{
            @as(f32, @floatFromInt(pixels[i].a)),
            @as(f32, @floatFromInt(pixels[i + 1].a)),
            @as(f32, @floatFromInt(pixels[i + 2].a)),
            @as(f32, @floatFromInt(pixels[i + 3].a)),
        };

        // Blend: result = original * (1-opacity) + gradient * opacity
        const blended_r_vec = orig_r_vec * one_minus_opacity_vec + grad_r_vec * opacity_vec;
        const blended_g_vec = orig_g_vec * one_minus_opacity_vec + grad_g_vec * opacity_vec;
        const blended_b_vec = orig_b_vec * one_minus_opacity_vec + grad_b_vec * opacity_vec;
        const blended_a_vec = orig_a_vec * one_minus_opacity_vec + grad_a_vec * opacity_vec;

        // Clamp and convert back to u8
        const clamped_r = simd.clampVec4F32ToU8(blended_r_vec);
        const clamped_g = simd.clampVec4F32ToU8(blended_g_vec);
        const clamped_b = simd.clampVec4F32ToU8(blended_b_vec);
        const clamped_a = simd.clampVec4F32ToU8(blended_a_vec);

        // Store back
        pixels[i] = img.color.Rgba32{ .r = clamped_r[0], .g = clamped_g[0], .b = clamped_b[0], .a = clamped_a[0] };
        pixels[i + 1] = img.color.Rgba32{ .r = clamped_r[1], .g = clamped_g[1], .b = clamped_b[1], .a = clamped_a[1] };
        pixels[i + 2] = img.color.Rgba32{ .r = clamped_r[2], .g = clamped_g[2], .b = clamped_b[2], .a = clamped_a[2] };
        pixels[i + 3] = img.color.Rgba32{ .r = clamped_r[3], .g = clamped_g[3], .b = clamped_b[3], .a = clamped_a[3] };
    }

    // Handle remaining pixels with scalar operations
    while (i < pixel_count) : (i += 1) {
        const y = i / width;
        const x = i % width;

        const dx = @as(f32, @floatFromInt(x)) - cx;
        const dy = @as(f32, @floatFromInt(y)) - cy;
        const distance = @sqrt(dx * dx + dy * dy);

        // Calculate interpolation factor (0 at center, 1 at max distance)
        const t = if (effective_radius > 0) std.math.clamp(distance / effective_radius, 0.0, 1.0) else 0.0;

        // Interpolate colors
        const r = @as(u8, @intFromFloat(@as(f32, @floatFromInt(start_rgba.r)) * (1.0 - t) + @as(f32, @floatFromInt(end_rgba.r)) * t));
        const g = @as(u8, @intFromFloat(@as(f32, @floatFromInt(start_rgba.g)) * (1.0 - t) + @as(f32, @floatFromInt(end_rgba.g)) * t));
        const b = @as(u8, @intFromFloat(@as(f32, @floatFromInt(start_rgba.b)) * (1.0 - t) + @as(f32, @floatFromInt(end_rgba.b)) * t));
        const a = @as(u8, @intFromFloat(@as(f32, @floatFromInt(start_rgba.a)) * (1.0 - t) + @as(f32, @floatFromInt(end_rgba.a)) * t));

        const original = pixels[i];

        // Blend with original image
        const blended_r = @as(u8, @intFromFloat(@as(f32, @floatFromInt(original.r)) * (1.0 - opacity) + @as(f32, @floatFromInt(r)) * opacity));
        const blended_g = @as(u8, @intFromFloat(@as(f32, @floatFromInt(original.g)) * (1.0 - opacity) + @as(f32, @floatFromInt(g)) * opacity));
        const blended_b = @as(u8, @intFromFloat(@as(f32, @floatFromInt(original.b)) * (1.0 - opacity) + @as(f32, @floatFromInt(b)) * opacity));
        const blended_a = @as(u8, @intFromFloat(@as(f32, @floatFromInt(original.a)) * (1.0 - opacity) + @as(f32, @floatFromInt(a)) * opacity));

        pixels[i].r = blended_r;
        pixels[i].g = blended_g;
        pixels[i].b = blended_b;
        pixels[i].a = blended_a;
    }

    utils.logMemoryUsage(ctx, "Radial gradient end");
}

pub fn censorImage(ctx: *Context, args: anytype) !void {
    const x = args[0]; // X coordinate of censor region (0.0-1.0)
    const y = args[1]; // Y coordinate of censor region (0.0-1.0)
    const width_pct = args[2]; // Width of censor region (0.0-1.0)
    const height_pct = args[3]; // Height of censor region (0.0-1.0)
    const method = args[4]; // "blur", "pixelate", or "black"
    const strength = args[5]; // Blur radius or pixelation block size

    // Input validation
    if (x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0) {
        std.log.err("Censor position must be in range 0.0-1.0, got ({d}, {d})", .{ x, y });
        return error.InvalidPosition;
    }
    if (width_pct <= 0.0 or width_pct > 1.0 or height_pct <= 0.0 or height_pct > 1.0) {
        std.log.err("Censor dimensions must be in range 0.0-1.0, got ({d}, {d})", .{ width_pct, height_pct });
        return error.InvalidDimensions;
    }
    if (!std.mem.eql(u8, method, "blur") and !std.mem.eql(u8, method, "pixelate") and !std.mem.eql(u8, method, "black")) {
        std.log.err("Censor method must be 'blur', 'pixelate', or 'black', got '{s}'", .{method});
        return error.InvalidMethod;
    }
    if (strength < 1) {
        std.log.err("Censor strength must be >= 1, got {d}", .{strength});
        return error.InvalidStrength;
    }

    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying censor effect at ({d:.2}, {d:.2}) size ({d:.2}, {d:.2}) with method '{s}' strength {d}", .{ x, y, width_pct, height_pct, method, strength });
    utils.logMemoryUsage(ctx, "Censor start");

    const pixels = ctx.image.pixels.rgba32;
    const img_width = ctx.image.width;
    const img_height = ctx.image.height;

    // Calculate censor region in pixel coordinates
    const start_x = @as(usize, @intFromFloat(x * @as(f32, @floatFromInt(img_width))));
    const start_y = @as(usize, @intFromFloat(y * @as(f32, @floatFromInt(img_height))));
    const region_width = @as(usize, @intFromFloat(width_pct * @as(f32, @floatFromInt(img_width))));
    const region_height = @as(usize, @intFromFloat(height_pct * @as(f32, @floatFromInt(img_height))));

    const end_x = @min(start_x + region_width, img_width);
    const end_y = @min(start_y + region_height, img_height);

    if (std.mem.eql(u8, method, "blur")) {
        // Apply blur to the region
        applyBlurToRegion(pixels, img_width, img_height, start_x, start_y, end_x, end_y, strength);
    } else if (std.mem.eql(u8, method, "pixelate")) {
        // Apply pixelation to the region
        applyPixelateToRegion(pixels, img_width, start_x, start_y, end_x, end_y, strength);
    } else if (std.mem.eql(u8, method, "black")) {
        // Fill region with black
        for (start_y..end_y) |py| {
            for (start_x..end_x) |px| {
                const idx = py * img_width + px;
                pixels[idx].r = 0;
                pixels[idx].g = 0;
                pixels[idx].b = 0;
                // Keep alpha
            }
        }
    }

    utils.logMemoryUsage(ctx, "Censor end");
}

/// Helper function to apply edge detection kernel for edge enhancement
fn applyEdgeDetectionKernel(pixels: []const img.color.Rgba32, output: []img.color.Rgba32, width: usize, height: usize) void {
    // Sobel edge detection kernel
    const sobel_x = [_]i32{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    const sobel_y = [_]i32{ -1, -2, -1, 0, 0, 0, 1, 2, 1 };

    for (0..height) |y| {
        for (0..width) |x| {
            var gx_r: i32 = 0;
            var gy_r: i32 = 0;
            var gx_g: i32 = 0;
            var gy_g: i32 = 0;
            var gx_b: i32 = 0;
            var gy_b: i32 = 0;

            // Apply 3x3 kernel
            for (0..3) |ky| {
                for (0..3) |kx| {
                    const nx_i32 = @as(i32, @intCast(x)) + @as(i32, @intCast(kx)) - 1;
                    const ny_i32 = @as(i32, @intCast(y)) + @as(i32, @intCast(ky)) - 1;

                    const clamped_x = std.math.clamp(nx_i32, 0, @as(i32, @intCast(width - 1)));
                    const clamped_y = std.math.clamp(ny_i32, 0, @as(i32, @intCast(height - 1)));
                    const idx = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x));

                    const kernel_idx = ky * 3 + kx;
                    const weight_x = sobel_x[kernel_idx];
                    const weight_y = sobel_y[kernel_idx];

                    gx_r += @as(i32, @intCast(pixels[idx].r)) * weight_x;
                    gy_r += @as(i32, @intCast(pixels[idx].r)) * weight_y;
                    gx_g += @as(i32, @intCast(pixels[idx].g)) * weight_x;
                    gy_g += @as(i32, @intCast(pixels[idx].g)) * weight_y;
                    gx_b += @as(i32, @intCast(pixels[idx].b)) * weight_x;
                    gy_b += @as(i32, @intCast(pixels[idx].b)) * weight_y;
                }
            }

            // Calculate gradient magnitude
            const mag_r = @sqrt(@as(f32, @floatFromInt(gx_r * gx_r + gy_r * gy_r)));
            const mag_g = @sqrt(@as(f32, @floatFromInt(gx_g * gx_g + gy_g * gy_g)));
            const mag_b = @sqrt(@as(f32, @floatFromInt(gx_b * gx_b + gy_b * gy_b)));

            const idx = y * width + x;
            output[idx].r = @as(u8, @intFromFloat(std.math.clamp(mag_r, 0.0, 255.0)));
            output[idx].g = @as(u8, @intFromFloat(std.math.clamp(mag_g, 0.0, 255.0)));
            output[idx].b = @as(u8, @intFromFloat(std.math.clamp(mag_b, 0.0, 255.0)));
            output[idx].a = 255;
        }
    }
}

// parseHexColor moved to utils.parseHexColor

/// Helper function to apply blur to a specific region
fn applyBlurToRegion(pixels: []img.color.Rgba32, width: usize, height: usize, start_x: usize, start_y: usize, end_x: usize, end_y: usize, radius: usize) void {
    // Simple box blur for the region
    const kernel_size = radius * 2 + 1;

    for (start_y..end_y) |y| {
        for (start_x..end_x) |x| {
            var r_sum: u32 = 0;
            var g_sum: u32 = 0;
            var b_sum: u32 = 0;
            var count: u32 = 0;

            // Apply kernel
            for (0..kernel_size) |ky| {
                for (0..kernel_size) |kx| {
                    const nx_i32 = @as(i32, @intCast(x)) + @as(i32, @intCast(kx)) - @as(i32, @intCast(radius));
                    const ny_i32 = @as(i32, @intCast(y)) + @as(i32, @intCast(ky)) - @as(i32, @intCast(radius));

                    if (nx_i32 >= 0 and nx_i32 < @as(i32, @intCast(width)) and ny_i32 >= 0 and ny_i32 < @as(i32, @intCast(height))) {
                        const nx = @as(usize, @intCast(nx_i32));
                        const ny = @as(usize, @intCast(ny_i32));
                        const idx = ny * width + nx;
                        r_sum += pixels[idx].r;
                        g_sum += pixels[idx].g;
                        b_sum += pixels[idx].b;
                        count += 1;
                    }
                }
            }

            const idx = y * width + x;
            if (count > 0) {
                pixels[idx].r = @as(u8, @intCast(r_sum / count));
                pixels[idx].g = @as(u8, @intCast(g_sum / count));
                pixels[idx].b = @as(u8, @intCast(b_sum / count));
            }
        }
    }
}

/// Helper function to apply pixelation to a specific region
fn applyPixelateToRegion(pixels: []img.color.Rgba32, width: usize, start_x: usize, start_y: usize, end_x: usize, end_y: usize, block_size: usize) void {
    var by = start_y;
    while (by < end_y) : (by += block_size) {
        var bx = start_x;
        while (bx < end_x) : (bx += block_size) {
            // Calculate average color for this block
            var sum_r: u32 = 0;
            var sum_g: u32 = 0;
            var sum_b: u32 = 0;
            var count: u32 = 0;

            const block_end_y = @min(by + block_size, end_y);
            const block_end_x = @min(bx + block_size, end_x);

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

/// Apply Sobel Gx kernel to a specific color channel
fn applySobelGxChannel(
    pixels: []const img.color.Rgba32,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    channel: Channel,
) i32 {
    var sum: i32 = 0;

    // Sobel Gx kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    sum += getChannelValue(pixels, width, height, x, y, -1, -1, channel) * (-1);
    sum += getChannelValue(pixels, width, height, x, y, 0, -1, channel) * 0;
    sum += getChannelValue(pixels, width, height, x, y, 1, -1, channel) * 1;
    sum += getChannelValue(pixels, width, height, x, y, -1, 0, channel) * (-2);
    sum += getChannelValue(pixels, width, height, x, y, 0, 0, channel) * 0;
    sum += getChannelValue(pixels, width, height, x, y, 1, 0, channel) * 2;
    sum += getChannelValue(pixels, width, height, x, y, -1, 1, channel) * (-1);
    sum += getChannelValue(pixels, width, height, x, y, 0, 1, channel) * 0;
    sum += getChannelValue(pixels, width, height, x, y, 1, 1, channel) * 1;

    return sum;
}

/// Apply Sobel Gy kernel to a specific color channel
fn applySobelGyChannel(
    pixels: []const img.color.Rgba32,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    channel: Channel,
) i32 {
    var sum: i32 = 0;

    // Sobel Gy kernel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
    sum += getChannelValue(pixels, width, height, x, y, -1, -1, channel) * (-1);
    sum += getChannelValue(pixels, width, height, x, y, 0, -1, channel) * (-2);
    sum += getChannelValue(pixels, width, height, x, y, 1, -1, channel) * (-1);
    sum += getChannelValue(pixels, width, height, x, y, -1, 0, channel) * 0;
    sum += getChannelValue(pixels, width, height, x, y, 0, 0, channel) * 0;
    sum += getChannelValue(pixels, width, height, x, y, 1, 0, channel) * 0;
    sum += getChannelValue(pixels, width, height, x, y, -1, 1, channel) * 1;
    sum += getChannelValue(pixels, width, height, x, y, 0, 1, channel) * 2;
    sum += getChannelValue(pixels, width, height, x, y, 1, 1, channel) * 1;

    return sum;
}
