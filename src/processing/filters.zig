const img = @import("zigimg");
const std = @import("std");
const Context = @import("../core/types.zig").Context;
const utils = @import("../core/utils.zig");
const math = std.math;

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

/// Apply 1D convolution horizontally
fn applyHorizontalConvolution(
    src_pixels: []const img.color.Rgba32,
    dst_pixels: []img.color.Rgba32,
    width: usize,
    height: usize,
    kernel: []const f32,
    kernel_radius: usize,
) void {
    const kernel_size = kernel_radius * 2 + 1;

    for (0..height) |y| {
        for (0..width) |x| {
            var r_sum: f32 = 0.0;
            var g_sum: f32 = 0.0;
            var b_sum: f32 = 0.0;
            var a_sum: f32 = 0.0;

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

/// Apply 1D convolution vertically
fn applyVerticalConvolution(
    src_pixels: []const img.color.Rgba32,
    dst_pixels: []img.color.Rgba32,
    width: usize,
    height: usize,
    kernel: []const f32,
    kernel_radius: usize,
) void {
    const kernel_size = kernel_radius * 2 + 1;

    for (0..height) |y| {
        for (0..width) |x| {
            var r_sum: f32 = 0.0;
            var g_sum: f32 = 0.0;
            var b_sum: f32 = 0.0;
            var a_sum: f32 = 0.0;

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

    // For separable blur, we need two temporary buffers
    // Use the context's temp buffer for the first pass, and allocate a second one
    const temp_pixels1 = try ctx.copyToTempBuffer(pixels);
    const temp_pixels2 = try ctx.allocator.alloc(img.color.Rgba32, pixels.len);
    defer ctx.allocator.free(temp_pixels2);

    // Apply separable blur: horizontal pass
    applyHorizontalConvolution(pixels, temp_pixels1, width, height, kernel, kernel_radius);

    // Apply separable blur: vertical pass
    applyVerticalConvolution(temp_pixels1, pixels, width, height, kernel, kernel_radius);

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

    // Simple sharpening using 3x3 kernel: center * 5 - orthogonal neighbors * 1
    // Skip 1-pixel border since kernel needs all neighbors
    if (height >= 3 and width >= 3) {
        for (1..height - 1) |y| {
            for (1..width - 1) |x| {
                const result = utils.applyKernel3x3i32(pixels, temp_pixels, width, height, x, y, [_][3]i32{
                    .{ 0, -1, 0 },
                    .{ -1, 5, -1 },
                    .{ 0, -1, 0 },
                });

                const idx = y * width + x;
                pixels[idx].r = utils.clampI16ToU8(@as(i16, @intCast(result.r)));
                pixels[idx].g = utils.clampI16ToU8(@as(i16, @intCast(result.g)));
                pixels[idx].b = utils.clampI16ToU8(@as(i16, @intCast(result.b)));
            }
        }
    }

    utils.logMemoryUsage(ctx, "Sharpen end");
}

/// Convert RGB pixel to grayscale luminance value
fn rgbToGrayLuminance(pixel: img.color.Rgba32) i32 {
    return @as(i32, @intCast(pixel.r)) * 77 +
        @as(i32, @intCast(pixel.g)) * 150 +
        @as(i32, @intCast(pixel.b)) * 29;
}

/// Get pixel value with edge replication for kernel operations
fn getPixelWithEdgeReplication(
    pixels: []const img.color.Rgba32,
    width: usize,
    height: usize,
    x: i32,
    y: i32,
) i32 {
    const clamped_x = std.math.clamp(x, 0, @as(i32, @intCast(width - 1)));
    const clamped_y = std.math.clamp(y, 0, @as(i32, @intCast(height - 1)));
    const idx = @as(usize, @intCast(clamped_y)) * width + @as(usize, @intCast(clamped_x));
    return rgbToGrayLuminance(pixels[idx]);
}

/// Apply emboss kernel to a single pixel with edge replication
fn applyEmbossKernel(
    pixels: []const img.color.Rgba32,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) i32 {
    var sum: i32 = 0;

    // Emboss kernel weights: [-2, -1, 0; -1, 1, 1; 0, 1, 2]
    // Top-left: -2
    sum += getPixelWithEdgeReplication(pixels, width, height, @as(i32, @intCast(x)) - 1, @as(i32, @intCast(y)) - 1) * (-2);
    // Top: -1
    sum += getPixelWithEdgeReplication(pixels, width, height, @as(i32, @intCast(x)), @as(i32, @intCast(y)) - 1) * (-1);
    // Left: -1
    sum += getPixelWithEdgeReplication(pixels, width, height, @as(i32, @intCast(x)) - 1, @as(i32, @intCast(y))) * (-1);
    // Bottom-right: 2
    sum += getPixelWithEdgeReplication(pixels, width, height, @as(i32, @intCast(x)) + 1, @as(i32, @intCast(y)) + 1) * 2;
    // Bottom: 1
    sum += getPixelWithEdgeReplication(pixels, width, height, @as(i32, @intCast(x)), @as(i32, @intCast(y)) + 1) * 1;
    // Right: 1
    sum += getPixelWithEdgeReplication(pixels, width, height, @as(i32, @intCast(x)) + 1, @as(i32, @intCast(y))) * 1;

    // Center: 1
    const center_idx = y * width + x;
    sum += rgbToGrayLuminance(pixels[center_idx]) * 1;

    return sum;
}

pub fn embossImage(ctx: *Context, args: anytype) !void {
    _ = args;
    try utils.convertToRgba32(ctx);

    utils.logVerbose(ctx, "Applying emboss effect", .{});
    utils.logMemoryUsage(ctx, "Emboss start");

    const pixels = ctx.image.pixels.rgba32;
    const width = ctx.image.width;
    const height = ctx.image.height;

    // Create a temporary buffer
    const temp_pixels = try ctx.copyToTempBuffer(pixels);

    // Process all pixels
    for (0..height) |y| {
        for (0..width) |x| {
            const emboss_sum = applyEmbossKernel(temp_pixels, width, height, x, y);

            // Normalize and add 128 to create the emboss effect
            const embossed = @as(i32, @intFromFloat(@as(f32, @floatFromInt(emboss_sum)) / 1024.0)) + 128;
            const clamped = std.math.clamp(embossed, 0, 255);
            const gray_value = @as(u8, @intCast(clamped));

            const idx = y * width + x;
            pixels[idx].r = gray_value;
            pixels[idx].g = gray_value;
            pixels[idx].b = gray_value;
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

    const center_x = @as(f32, @floatFromInt(width)) / 2.0;
    const center_y = @as(f32, @floatFromInt(height)) / 2.0;
    const max_distance = @sqrt(center_x * center_x + center_y * center_y);
    const inv_max_distance = 1.0 / max_distance;

    for (0..height) |y| {
        const dy = @as(f32, @floatFromInt(y)) - center_y;
        const dy_sq = dy * dy;

        for (0..width) |x| {
            const dx = @as(f32, @floatFromInt(x)) - center_x;
            const distance = @sqrt(dx * dx + dy_sq);
            const normalized_distance = distance * inv_max_distance;

            // Vignette factor: closer to center = 1.0, edges = intensity
            const vignette_factor = 1.0 - (normalized_distance * intensity);

            const idx = y * width + x;
            pixels[idx].r = utils.clampU8(@as(f32, @floatFromInt(pixels[idx].r)) * vignette_factor);
            pixels[idx].g = utils.clampU8(@as(f32, @floatFromInt(pixels[idx].g)) * vignette_factor);
            pixels[idx].b = utils.clampU8(@as(f32, @floatFromInt(pixels[idx].b)) * vignette_factor);
        }
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
            const gx_result = utils.applyKernel3x3i32(pixels, temp_pixels, width, height, x, y, gx);
            const gy_result = utils.applyKernel3x3i32(pixels, temp_pixels, width, height, x, y, gy);

            // Calculate magnitude
            const magnitude_r = @sqrt(@as(f32, @floatFromInt(gx_result.r * gx_result.r + gy_result.r * gy_result.r)));
            const magnitude_g = @sqrt(@as(f32, @floatFromInt(gx_result.g * gx_result.g + gy_result.g * gy_result.g)));
            const magnitude_b = @sqrt(@as(f32, @floatFromInt(gx_result.b * gx_result.b + gy_result.b * gy_result.b)));

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

        tile_start_y = tile_end_y;
    }

    utils.logMemoryUsage(ctx, "Oil painting end");
}
