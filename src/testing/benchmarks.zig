const std = @import("std");
const build_options = @import("build_options");
const img = @import("zigimg");
const color = @import("../processing/color.zig");
const filters = @import("../processing/filters.zig");
const transforms = @import("../processing/transforms.zig");
const types = @import("../core/types.zig");

/// Create a test image with specified dimensions and color
fn createBenchmarkImage(allocator: std.mem.Allocator, width: usize, height: usize, pixel_color: img.color.Rgba32) !img.Image {
    const image = try img.Image.create(allocator, width, height, .rgba32);
    for (image.pixels.rgba32) |*pixel| {
        pixel.* = pixel_color;
    }
    return image;
}

/// Benchmark utility function
fn benchmarkOperation(
    allocator: std.mem.Allocator,
    comptime operation_name: []const u8,
    operation: fn (*types.Context, anytype) anyerror!void,
    args: anytype,
    width: usize,
    height: usize,
    iterations: usize,
) !void {
    // Warm up
    {
        var ctx = types.Context.init(allocator);
        defer ctx.deinit();

        const image = try createBenchmarkImage(allocator, width, height, img.color.Rgba32{ .r = 128, .g = 128, .b = 128, .a = 255 });
        ctx.setImage(image);

        // Run operation once to warm up
        try operation(&ctx, args);
    }

    // Benchmark
    var total_time: u64 = 0;
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;

    for (0..iterations) |_| {
        var ctx = types.Context.init(allocator);
        defer ctx.deinit();

        const image = try createBenchmarkImage(allocator, width, height, img.color.Rgba32{ .r = 128, .g = 128, .b = 128, .a = 255 });
        ctx.setImage(image);

        const start = std.time.nanoTimestamp();
        try operation(&ctx, args);
        const end = std.time.nanoTimestamp();

        const duration = @as(u64, @intCast(end - start));
        total_time += duration;
        min_time = @min(min_time, duration);
        max_time = @max(max_time, duration);
    }

    const avg_time = total_time / iterations;
    const pixels_processed = width * height * iterations;
    const pixels_per_second = @as(f64, @floatFromInt(pixels_processed)) / (@as(f64, @floatFromInt(avg_time)) / 1_000_000_000.0);

    std.debug.print("{s} ({d}x{d}, {d} iterations):\n", .{ operation_name, width, height, iterations });
    std.debug.print("  Avg: {d:.2} ms\n", .{@as(f64, @floatFromInt(avg_time)) / 1_000_000.0});
    std.debug.print("  Min: {d:.2} ms\n", .{@as(f64, @floatFromInt(min_time)) / 1_000_000.0});
    std.debug.print("  Max: {d:.2} ms\n", .{@as(f64, @floatFromInt(max_time)) / 1_000_000.0});
    std.debug.print("  Throughput: {d:.0} MPixels/sec\n\n", .{pixels_per_second / 1_000_000.0});
}

pub fn runBenchmarks(allocator: std.mem.Allocator) !void {
    std.debug.print("Running mimg Performance Benchmarks\n", .{});
    std.debug.print("===================================\n\n", .{});

    // Test different image sizes
    const sizes = [_][2]usize{
        .{ 256, 256 }, // Small
        .{ 1024, 1024 }, // Medium
        .{ 2048, 1024 }, // Large (2K)
    };

    const iterations = 10;

    for (sizes) |size| {
        const width = size[0];
        const height = size[1];

        std.debug.print("Benchmarking {d}x{d} images:\n", .{ width, height });
        std.debug.print("------------------------\n", .{});

        // Color operations
        try benchmarkOperation(allocator, "Grayscale", color.grayscaleImage, .{}, width, height, iterations);
        try benchmarkOperation(allocator, "Invert Colors", color.invertColors, .{}, width, height, iterations);
        try benchmarkOperation(allocator, "Adjust Brightness (+50)", color.adjustBrightness, .{50}, width, height, iterations);
        try benchmarkOperation(allocator, "Adjust Contrast (2.0x)", color.adjustContrast, .{2.0}, width, height, iterations);
        try benchmarkOperation(allocator, "Apply Sepia", color.applySepia, .{}, width, height, iterations);

        // Filter operations
        try benchmarkOperation(allocator, "Box Blur (3x3)", filters.blurImage, .{3}, width, height, iterations);
        try benchmarkOperation(allocator, "Gaussian Blur (sigma=1.0)", filters.gaussianBlurImage, .{1.0}, width, height, iterations);
        try benchmarkOperation(allocator, "Sharpen", filters.sharpenImage, .{1.0}, width, height, iterations);
        try benchmarkOperation(allocator, "Median Filter (3x3)", filters.medianFilterImage, .{3}, width, height, iterations);

        // Transform operations
        try benchmarkOperation(allocator, "Flip Horizontal", transforms.flipImage, .{"horizontal"}, width, height, iterations);
        try benchmarkOperation(allocator, "Rotate 90 degrees", transforms.rotateImage, .{90}, width, height, iterations);

        std.debug.print("\n", .{});
    }
}

test "run performance benchmarks" {
    const allocator = std.testing.allocator;
    try runBenchmarks(allocator);
}
