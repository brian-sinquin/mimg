const std = @import("std");
const testing = std.testing;
const utils = @import("utils.zig");
const types = @import("types.zig");
const basic = @import("basic.zig");
const img = @import("zigimg");

// Test utilities
fn createTestImage(allocator: std.mem.Allocator, width: usize, height: usize, color: img.color.Rgba32) !img.Image {
    const image = try img.Image.create(allocator, width, height, .rgba32);
    for (image.pixels.rgba32) |*pixel| {
        pixel.* = color;
    }
    return image;
}

fn createGradientImage(allocator: std.mem.Allocator, width: usize, height: usize) !img.Image {
    const image = try img.Image.create(allocator, width, height, .rgba32);
    for (image.pixels.rgba32, 0..) |*pixel, i| {
        const x = i % width;
        const y = i / width;
        const r = @as(u8, @intCast((x * 255) / width));
        const g = @as(u8, @intCast((y * 255) / height));
        pixel.* = img.color.Rgba32{ .r = r, .g = g, .b = 128, .a = 255 };
    }
    return image;
}

test "resolveOutputPath - extension handling" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    ctx.input_filename = try allocator.dupe(u8, "input.jpg");
    var buffer: [1024]u8 = undefined;

    // Test extension preservation
    ctx.setOutput("processed.jpg");
    const path = try utils.resolveOutputPath(&ctx, "out", &buffer);
    try testing.expectEqualStrings("processed.jpg", path);
    try testing.expectEqualStrings(".jpg", ctx.output_extension);
}

// Utility function tests
test "parseArgsFromSlice - basic types" {
    var arg_index: usize = 0;

    // Test string parsing
    const string_args = &[_][]const u8{ "hello", "world" };
    const string_result = try utils.parseArgsFromSlice(&[_]type{[]const u8}, string_args, &arg_index);
    try testing.expectEqualStrings("hello", string_result[0]);
    try testing.expectEqual(@as(usize, 1), arg_index);

    // Test integer parsing
    arg_index = 0;
    const int_args = &[_][]const u8{ "42", "-10" };
    const int_result = try utils.parseArgsFromSlice(&[_]type{ i32, i32 }, int_args, &arg_index);
    try testing.expectEqual(@as(i32, 42), int_result[0]);
    try testing.expectEqual(@as(i32, -10), int_result[1]);
    try testing.expectEqual(@as(usize, 2), arg_index);

    // Test float parsing
    arg_index = 0;
    const float_args = &[_][]const u8{ "3.14", "-2.5" };
    const float_result = try utils.parseArgsFromSlice(&[_]type{ f32, f64 }, float_args, &arg_index);
    try testing.expectApproxEqAbs(@as(f32, 3.14), float_result[0], 0.001);
    try testing.expectApproxEqAbs(@as(f64, -2.5), float_result[1], 0.001);
    try testing.expectEqual(@as(usize, 2), arg_index);
}

test "parseArgsFromSlice - error handling" {
    var arg_index: usize = 0;

    // Test missing argument
    const short_args = &[_][]const u8{"42"};
    const result = utils.parseArgsFromSlice(&[_]type{ i32, i32 }, short_args, &arg_index);
    try testing.expectError(types.ParseArgError.MissingArgument, result);

    // Test invalid number
    arg_index = 0;
    const invalid_args = &[_][]const u8{"not_a_number"};
    const invalid_result = utils.parseArgsFromSlice(&[_]type{i32}, invalid_args, &arg_index);
    try testing.expectError(types.ParseArgError.InvalidArgument, invalid_result);
}

test "resolveOutputPath - basic functionality" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    ctx.input_filename = try allocator.dupe(u8, "test.png");
    var buffer: [1024]u8 = undefined;

    // Test with no modifiers (output_filename is "out")
    const path1 = try utils.resolveOutputPath(&ctx, "out", &buffer);
    try testing.expectEqualStrings("test.png", path1);

    // Test with modifiers
    ctx.setOutput("brightness_50");
    const path2 = try utils.resolveOutputPath(&ctx, "out", &buffer);
    try testing.expectEqualStrings("brightness_50_test.png", path2);

    // Test with output directory
    ctx.setOutputDirectory("output");
    const path3 = try utils.resolveOutputPath(&ctx, "out", &buffer);
    const expected_path = try std.fs.path.join(std.testing.allocator, &[_][]const u8{ "output", "brightness_50_test.png" });
    defer std.testing.allocator.free(expected_path);
    try testing.expectEqualStrings(expected_path, path3);
}

test "expandWildcard - basic patterns" {
    var allocator = std.testing.allocator;

    // Create a temporary directory for the test
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const test_files = [_][]const u8{ "test1.png", "test2.png", "other.jpg" };
    for (test_files) |filename| {
        const file = try tmp_dir.dir.createFile(filename, .{});
        file.close();
    }

    // Change to the temporary directory for this test
    const original_cwd = std.fs.cwd();
    try tmp_dir.dir.setAsCwd();
    defer original_cwd.setAsCwd() catch {};

    // Test wildcard expansion
    const result = try utils.expandWildcard(allocator, "test*.png");
    defer {
        for (result) |path| {
            allocator.free(path);
        }
        allocator.free(result);
    }

    try testing.expectEqual(@as(usize, 2), result.len);
    try testing.expect(std.mem.eql(u8, result[0], "test1.png") or std.mem.eql(u8, result[0], "test2.png"));
    try testing.expect(std.mem.eql(u8, result[1], "test1.png") or std.mem.eql(u8, result[1], "test2.png"));
}

test "getExtensionFromSource - various formats" {
    // Test local files
    try testing.expectEqualStrings(".png", utils.getExtensionFromSource("image.png"));
    try testing.expectEqualStrings(".jpg", utils.getExtensionFromSource("photo.jpg"));
    try testing.expectEqualStrings(".png", utils.getExtensionFromSource("file_without_ext"));

    // Test URLs
    try testing.expectEqualStrings(".jpg", utils.getExtensionFromSource("https://example.com/image.jpg"));
    try testing.expectEqualStrings(".png", utils.getExtensionFromSource("https://example.com/image.png?param=value"));
    try testing.expectEqualStrings(".png", utils.getExtensionFromSource("https://example.com/image"));
}

test "isValidUrl - validation" {
    try testing.expect(utils.isValidUrl("https://example.com/image.png"));
    try testing.expect(utils.isValidUrl("http://example.com/image.jpg"));
    try testing.expect(!utils.isValidUrl("ftp://example.com/image.png"));
    try testing.expect(!utils.isValidUrl("file:///path/to/image.png"));
    try testing.expect(!utils.isValidUrl("not_a_url"));
}

// Image processing tests
test "invertColors - basic functionality" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test image with known colors
    const image = try createTestImage(allocator, 2, 2, img.color.Rgba32{ .r = 100, .g = 150, .b = 200, .a = 255 });
    ctx.setImage(image);

    // Apply invert
    try basic.invertColors(&ctx, .{});

    // Check results
    const pixels = ctx.image.pixels.rgba32;
    try testing.expectEqual(@as(u8, 155), pixels[0].r); // 255 - 100
    try testing.expectEqual(@as(u8, 105), pixels[0].g); // 255 - 150
    try testing.expectEqual(@as(u8, 55), pixels[0].b); // 255 - 200
    try testing.expectEqual(@as(u8, 255), pixels[0].a); // Alpha unchanged
}

test "grayscaleImage - luminance calculation" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test image
    const image = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 255, .g = 128, .b = 0, .a = 255 });
    ctx.setImage(image);

    // Apply grayscale
    try basic.grayscaleImage(&ctx, .{});

    // Check result (should be luminance: 0.299*255 + 0.587*128 + 0.114*0 ≈ 151)
    const pixel = ctx.image.pixels.rgba32[0];
    try testing.expectEqual(@as(u8, 151), pixel.r);
    try testing.expectEqual(@as(u8, 151), pixel.g);
    try testing.expectEqual(@as(u8, 151), pixel.b);
    try testing.expectEqual(@as(u8, 255), pixel.a);
}

test "adjustBrightness - positive and negative" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Test positive brightness
    const image1 = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 100, .g = 100, .b = 100, .a = 255 });
    ctx.setImage(image1);
    try basic.adjustBrightness(&ctx, .{50});

    const pixel1 = ctx.image.pixels.rgba32[0];
    try testing.expectEqual(@as(u8, 150), pixel1.r);
    try testing.expectEqual(@as(u8, 150), pixel1.g);
    try testing.expectEqual(@as(u8, 150), pixel1.b);

    // Test negative brightness (clamping)
    const image2 = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 50, .g = 50, .b = 50, .a = 255 });
    ctx.setImage(image2);
    try basic.adjustBrightness(&ctx, .{-100});

    const pixel2 = ctx.image.pixels.rgba32[0];
    try testing.expectEqual(@as(u8, 0), pixel2.r);
    try testing.expectEqual(@as(u8, 0), pixel2.g);
    try testing.expectEqual(@as(u8, 0), pixel2.b);
}

test "adjustContrast - basic functionality" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a gradient image for testing contrast
    const image = try createGradientImage(allocator, 3, 1);
    ctx.setImage(image);

    // Apply contrast adjustment
    try basic.adjustContrast(&ctx, .{2.0});

    // The middle value should be more extreme
    const pixels = ctx.image.pixels.rgba32;
    try testing.expect(pixels[1].r > pixels[0].r); // Increased contrast
}

test "resizeImage - nearest neighbor" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a 2x2 test image
    const image = try img.Image.create(allocator, 2, 2, .rgba32);
    image.pixels.rgba32[0] = img.color.Rgba32{ .r = 255, .g = 0, .b = 0, .a = 255 }; // Red
    image.pixels.rgba32[1] = img.color.Rgba32{ .r = 0, .g = 255, .b = 0, .a = 255 }; // Green
    image.pixels.rgba32[2] = img.color.Rgba32{ .r = 0, .g = 0, .b = 255, .a = 255 }; // Blue
    image.pixels.rgba32[3] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 }; // White
    ctx.setImage(image);

    // Resize to 1x1
    try basic.resizeImage(&ctx, .{ 1, 1 });

    try testing.expectEqual(@as(usize, 1), ctx.image.width);
    try testing.expectEqual(@as(usize, 1), ctx.image.height);
    // Should sample from top-left corner (red pixel)
    try testing.expectEqual(@as(u8, 255), ctx.image.pixels.rgba32[0].r);
    try testing.expectEqual(@as(u8, 0), ctx.image.pixels.rgba32[0].g);
}

test "cropImage - basic functionality" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a 4x4 test image
    const image = try img.Image.create(allocator, 4, 4, .rgba32);
    for (image.pixels.rgba32, 0..) |*pixel, i| {
        pixel.* = img.color.Rgba32{ .r = @as(u8, @intCast(i)), .g = @as(u8, @intCast(i)), .b = @as(u8, @intCast(i)), .a = 255 };
    }
    ctx.setImage(image);

    // Crop a 2x2 region starting at (1,1)
    try basic.cropImage(&ctx, .{ 1, 1, 2, 2 });

    try testing.expectEqual(@as(usize, 2), ctx.image.width);
    try testing.expectEqual(@as(usize, 2), ctx.image.height);

    // Check that we got the right pixels
    const pixels = ctx.image.pixels.rgba32;
    try testing.expectEqual(@as(u8, 5), pixels[0].r); // Index 5 (1,1) in original
    try testing.expectEqual(@as(u8, 6), pixels[1].r); // Index 6 (2,1) in original
    try testing.expectEqual(@as(u8, 9), pixels[2].r); // Index 9 (1,2) in original
    try testing.expectEqual(@as(u8, 10), pixels[3].r); // Index 10 (2,2) in original
}

// Context tests
test "Context - basic operations" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Test initial state
    try testing.expect(!ctx.image_loaded);
    try testing.expectEqualStrings("out", ctx.output_filename);
    try testing.expectEqualStrings(".png", ctx.output_extension);

    // Test output filename setting
    ctx.setOutput("processed");
    try testing.expectEqualStrings("processed", ctx.output_filename);

    // Test output directory setting
    ctx.setOutputDirectory("output_dir");
    try testing.expectEqualStrings("output_dir", ctx.output_directory.?);

    // Test verbose setting
    ctx.setVerbose(true);
    try testing.expect(ctx.verbose);
}

test "Context - setOutput with extension" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Test filename with extension
    ctx.setOutput("output.jpg");
    try testing.expectEqualStrings("output.jpg", ctx.output_filename);
    try testing.expectEqualStrings(".jpg", ctx.output_extension);

    // Test filename without extension
    ctx.setOutput("output2");
    try testing.expectEqualStrings("output2", ctx.output_filename);
    try testing.expectEqualStrings(".jpg", ctx.output_extension); // Should retain previous extension
}

// Pixel utility tests
test "clampU8 - boundary values" {
    try testing.expectEqual(@as(u8, 0), utils.clampU8(-1.0));
    try testing.expectEqual(@as(u8, 0), utils.clampU8(0.0));
    try testing.expectEqual(@as(u8, 128), utils.clampU8(128.0));
    try testing.expectEqual(@as(u8, 255), utils.clampU8(255.0));
    try testing.expectEqual(@as(u8, 255), utils.clampU8(300.0));
}

test "rgbToLuminance - standard values" {
    try testing.expectApproxEqAbs(@as(f32, 76.245), utils.rgbToLuminance(255, 0, 0), 0.001); // Red
    try testing.expectApproxEqAbs(@as(f32, 149.685), utils.rgbToLuminance(0, 255, 0), 0.001); // Green
    try testing.expectApproxEqAbs(@as(f32, 29.07), utils.rgbToLuminance(0, 0, 255), 0.001); // Blue
    try testing.expectApproxEqAbs(@as(f32, 255.0), utils.rgbToLuminance(255, 255, 255), 0.001); // White
    try testing.expectApproxEqAbs(@as(f32, 0.0), utils.rgbToLuminance(0, 0, 0), 0.001); // Black
}

test "getPixelSafe - bounds checking" {
    const allocator = std.testing.allocator;
    var pixels = try allocator.alloc(img.color.Rgba32, 4);
    defer allocator.free(pixels);

    pixels[0] = img.color.Rgba32{ .r = 255, .g = 0, .b = 0, .a = 255 };
    pixels[1] = img.color.Rgba32{ .r = 0, .g = 255, .b = 0, .a = 255 };
    pixels[2] = img.color.Rgba32{ .r = 0, .g = 0, .b = 255, .a = 255 };
    pixels[3] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 };

    // Valid pixels
    const pixel00 = utils.getPixelSafe(pixels, 2, 2, 0, 0);
    try testing.expect(pixel00 != null);
    try testing.expectEqual(@as(u8, 255), pixel00.?.r);

    const pixel11 = utils.getPixelSafe(pixels, 2, 2, 1, 1);
    try testing.expect(pixel11 != null);
    try testing.expectEqual(@as(u8, 255), pixel11.?.g);

    // Out of bounds
    try testing.expectEqual(@as(?img.color.Rgba32, null), utils.getPixelSafe(pixels, 2, 2, -1, 0));
    try testing.expectEqual(@as(?img.color.Rgba32, null), utils.getPixelSafe(pixels, 2, 2, 0, -1));
    try testing.expectEqual(@as(?img.color.Rgba32, null), utils.getPixelSafe(pixels, 2, 2, 2, 0));
    try testing.expectEqual(@as(?img.color.Rgba32, null), utils.getPixelSafe(pixels, 2, 2, 0, 2));
}

test "getPixelClamped - edge behavior" {
    const allocator = std.testing.allocator;
    var pixels = try allocator.alloc(img.color.Rgba32, 4);
    defer allocator.free(pixels);

    pixels[0] = img.color.Rgba32{ .r = 255, .g = 0, .b = 0, .a = 255 };
    pixels[1] = img.color.Rgba32{ .r = 0, .g = 255, .b = 0, .a = 255 };
    pixels[2] = img.color.Rgba32{ .r = 0, .g = 0, .b = 255, .a = 255 };
    pixels[3] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 };

    // In bounds
    const pixel00 = utils.getPixelClamped(pixels, 2, 2, 0, 0);
    try testing.expectEqual(@as(u8, 255), pixel00.r);

    // Out of bounds (should clamp to edges)
    const pixel_neg = utils.getPixelClamped(pixels, 2, 2, -1, -1);
    try testing.expectEqual(@as(u8, 255), pixel_neg.r); // Should get pixel[0]

    const pixel_over = utils.getPixelClamped(pixels, 2, 2, 2, 2);
    try testing.expectEqual(@as(u8, 255), pixel_over.r); // Should get pixel[3]
    try testing.expectEqual(@as(u8, 255), pixel_over.g);
    try testing.expectEqual(@as(u8, 255), pixel_over.b);
}

test "adjustSaturation - basic functionality" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test image with medium saturation
    const image = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 128, .g = 64, .b = 192, .a = 255 });
    ctx.setImage(image);

    // Increase saturation
    try basic.adjustSaturation(&ctx, .{1.5});

    const pixel = ctx.image.pixels.rgba32[0];
    // Saturation increase should make colors more vivid
    try testing.expect(pixel.r >= 128); // Red should increase
    try testing.expect(pixel.g <= 64); // Green should decrease
    try testing.expect(pixel.b >= 192); // Blue should increase
}

test "adjustGamma - basic functionality" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test image
    const image = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 128, .g = 128, .b = 128, .a = 255 });
    ctx.setImage(image);

    // Apply gamma correction (gamma < 1 makes dark areas brighter but midtones darker)
    try basic.adjustGamma(&ctx, .{0.5});

    const pixel = ctx.image.pixels.rgba32[0];
    // Gamma < 1 should make mid-tones darker
    try testing.expect(pixel.r < 128);
    try testing.expect(pixel.g < 128);
    try testing.expect(pixel.b < 128);
}

test "adjustExposure - basic functionality" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test image
    const image = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 128, .g = 128, .b = 128, .a = 255 });
    ctx.setImage(image);

    // Increase exposure
    try basic.adjustExposure(&ctx, .{0.5});

    const pixel = ctx.image.pixels.rgba32[0];
    // Exposure increase should brighten the image
    try testing.expect(pixel.r > 128);
    try testing.expect(pixel.g > 128);
    try testing.expect(pixel.b > 128);
}

test "adjustVibrance - basic functionality" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a colorful test image
    const image = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 200, .g = 100, .b = 50, .a = 255 });
    ctx.setImage(image);

    // Increase vibrance
    try basic.adjustVibrance(&ctx, .{0.3});

    const pixel = ctx.image.pixels.rgba32[0];
    // Vibrance should adjust colors (less saturated colors boosted more)
    // With input (200,100,50), green should decrease slightly
    try testing.expect(pixel.g < 100);
    try testing.expect(pixel.r >= 200);
    try testing.expect(pixel.b <= 50);
}

test "equalizeImage - histogram equalization" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a gradient image for testing equalization
    const image = try createGradientImage(allocator, 256, 1);
    ctx.setImage(image);

    // Apply histogram equalization
    try basic.equalizeImage(&ctx, .{});

    // After equalization, the image should have better contrast
    // Check that we have both dark and light pixels
    const pixels = ctx.image.pixels.rgba32;
    var has_dark = false;
    var has_light = false;
    for (pixels) |pixel| {
        if (pixel.r < 64) has_dark = true;
        if (pixel.r > 192) has_light = true;
    }
    try testing.expect(has_dark and has_light);
}

test "hueShiftImage - color rotation" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a pure red image
    const image = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 255, .g = 0, .b = 0, .a = 255 });
    ctx.setImage(image);

    // Shift hue by 120 degrees (should become green)
    try basic.hueShiftImage(&ctx, .{120.0});

    const pixel = ctx.image.pixels.rgba32[0];
    // Should be approximately green now
    try testing.expect(pixel.g > 200);
    try testing.expect(pixel.r < 50);
    try testing.expect(pixel.b < 50);
}

test "applySepia - sepia tone effect" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test image
    const image = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 });
    ctx.setImage(image);

    // Apply sepia effect
    try basic.applySepia(&ctx, .{});

    const pixel = ctx.image.pixels.rgba32[0];
    // Sepia should give warm brownish tones (r and g high, b lower)
    try testing.expect(pixel.r >= 240);
    try testing.expect(pixel.g >= 240);
    try testing.expect(pixel.b < 240);
}

test "colorizeImage - single color tint" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a grayscale image
    const image = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 128, .g = 128, .b = 128, .a = 255 });
    ctx.setImage(image);

    // Colorize with red and intensity
    try basic.colorizeImage(&ctx, .{ 255, 0, 0, 0.5 });

    const pixel = ctx.image.pixels.rgba32[0];
    // Should be tinted red
    try testing.expect(pixel.r > pixel.g);
    try testing.expect(pixel.r > pixel.b);
}

test "duotoneImage - two color gradient" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a gradient image
    const image = try createGradientImage(allocator, 3, 1);
    ctx.setImage(image);

    // Apply duotone effect
    try basic.duotoneImage(&ctx, .{ 255, 0, 0, 0, 0, 255 }); // Red to blue

    const pixels = ctx.image.pixels.rgba32;
    // First pixel should be more red (dark color)
    try testing.expect(pixels[0].r > pixels[0].b);
    // Last pixel should be more red than blue (low luminance)
    try testing.expect(pixels[2].r > pixels[2].b);
}

test "thresholdImage - binary conversion" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a gradient image
    const image = try createGradientImage(allocator, 3, 1);
    ctx.setImage(image);

    // Apply threshold
    try basic.thresholdImage(&ctx, .{128});

    const pixels = ctx.image.pixels.rgba32;
    // Should be binary: black or white
    for (pixels) |pixel| {
        const is_black = pixel.r == 0 and pixel.g == 0 and pixel.b == 0;
        const is_white = pixel.r == 255 and pixel.g == 255 and pixel.b == 255;
        try testing.expect(is_black or is_white);
    }
}

test "solarizeImage - solarization effect" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a bright image
    const image = try createTestImage(allocator, 1, 1, img.color.Rgba32{ .r = 200, .g = 200, .b = 200, .a = 255 });
    ctx.setImage(image);

    // Apply solarization
    try basic.solarizeImage(&ctx, .{128});

    const pixel = ctx.image.pixels.rgba32[0];
    // Solarization should invert bright areas
    try testing.expect(pixel.r < 128);
    try testing.expect(pixel.g < 128);
    try testing.expect(pixel.b < 128);
}

test "posterizeImage - reduce color levels" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a gradient image
    const image = try createGradientImage(allocator, 256, 1);
    ctx.setImage(image);

    // Apply posterization (4 levels)
    try basic.posterizeImage(&ctx, .{4});

    const pixels = ctx.image.pixels.rgba32;
    // Check that colors are quantized to specific levels
    for (pixels) |pixel| {
        const valid_levels = [_]u8{ 0, 85, 170, 255 };
        var is_valid = false;
        for (valid_levels) |level| {
            if (pixel.r == level) is_valid = true;
        }
        try testing.expect(is_valid);
    }
}

// Filter tests
test "blurImage - box blur" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test pattern
    const image = try img.Image.create(allocator, 3, 3, .rgba32);
    // Create a checkerboard pattern
    image.pixels.rgba32[0] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 };
    image.pixels.rgba32[1] = img.color.Rgba32{ .r = 0, .g = 0, .b = 0, .a = 255 };
    image.pixels.rgba32[2] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 };
    image.pixels.rgba32[3] = img.color.Rgba32{ .r = 0, .g = 0, .b = 0, .a = 255 };
    image.pixels.rgba32[4] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 };
    image.pixels.rgba32[5] = img.color.Rgba32{ .r = 0, .g = 0, .b = 0, .a = 255 };
    image.pixels.rgba32[6] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 };
    image.pixels.rgba32[7] = img.color.Rgba32{ .r = 0, .g = 0, .b = 0, .a = 255 };
    image.pixels.rgba32[8] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 };
    ctx.setImage(image);

    // Apply blur
    try basic.blurImage(&ctx, .{3});

    const pixels = ctx.image.pixels.rgba32;
    // Center pixel should be averaged (5 white + 4 black = 141.67)
    try testing.expect(pixels[4].r > 0 and pixels[4].r < 255);
}

test "gaussianBlurImage - gaussian blur" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a simple test image
    const image = try createTestImage(allocator, 5, 5, img.color.Rgba32{ .r = 128, .g = 128, .b = 128, .a = 255 });
    // Add a bright spot in the center
    image.pixels.rgba32[12] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 };
    ctx.setImage(image);

    // Apply gaussian blur
    try basic.gaussianBlurImage(&ctx, .{1.0});

    const pixels = ctx.image.pixels.rgba32;
    // Center should be blurred but still brighter than edges
    try testing.expect(pixels[12].r > pixels[0].r);
    try testing.expect(pixels[12].r < 255);
}

test "sharpenImage - unsharp mask" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test image with edges
    const image = try createGradientImage(allocator, 3, 1);
    ctx.setImage(image);

    // Apply sharpen
    try basic.sharpenImage(&ctx, .{1.0});

    const pixels = ctx.image.pixels.rgba32;
    // Edges should be enhanced (center becomes more extreme)
    try testing.expect(pixels[1].r != 128); // Middle should change
}

test "embossImage - emboss effect" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a gradient for emboss effect
    const image = try createGradientImage(allocator, 3, 1);
    ctx.setImage(image);

    // Apply emboss
    try basic.embossImage(&ctx, .{});

    const pixels = ctx.image.pixels.rgba32;
    // Emboss should create light/dark edges
    try testing.expect(pixels[1].r != 128); // Middle should change
}

test "vignetteImage - corner darkening" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test image
    const image = try createTestImage(allocator, 10, 10, img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 });
    ctx.setImage(image);

    // Apply vignette
    try basic.vignetteImage(&ctx, .{0.5});

    const pixels = ctx.image.pixels.rgba32;
    // Corners should be darker than center
    const corner = pixels[0];
    const center = pixels[55]; // Approximate center
    try testing.expect(center.r > corner.r);
}

test "edgeDetectImage - sobel operator" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create an image with a clear edge
    const image = try img.Image.create(allocator, 3, 3, .rgba32);
    // Left half black, right half white
    for (image.pixels.rgba32, 0..) |*pixel, i| {
        const x = i % 3;
        if (x < 2) {
            pixel.* = img.color.Rgba32{ .r = 0, .g = 0, .b = 0, .a = 255 };
        } else {
            pixel.* = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 };
        }
    }
    ctx.setImage(image);

    // Apply edge detection
    try basic.edgeDetectImage(&ctx, .{});

    const pixels = ctx.image.pixels.rgba32;
    // Should detect the vertical edge
    try testing.expect(pixels[4].r > 0); // Center should show edge
}

test "medianFilterImage - noise reduction" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create an image with some noise
    const image = try img.Image.create(allocator, 3, 3, .rgba32);
    for (image.pixels.rgba32) |*pixel| {
        pixel.* = img.color.Rgba32{ .r = 128, .g = 128, .b = 128, .a = 255 };
    }
    // Add noise to center
    image.pixels.rgba32[4] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 };
    ctx.setImage(image);

    // Apply median filter
    try basic.medianFilterImage(&ctx, .{3});

    const pixels = ctx.image.pixels.rgba32;
    // Center should be closer to surrounding pixels (median of 128,128,128,128,255,128,128,128,128 = 128)
    try testing.expect(pixels[4].r < 255);
}

test "addNoiseImage - noise addition" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a uniform image
    const image = try createTestImage(allocator, 10, 10, img.color.Rgba32{ .r = 128, .g = 128, .b = 128, .a = 255 });
    ctx.setImage(image);

    // Add noise
    try basic.addNoiseImage(&ctx, .{0.25});

    const pixels = ctx.image.pixels.rgba32;
    // Should have variation now
    var has_variation = false;
    for (pixels) |pixel| {
        if (pixel.r != 128) has_variation = true;
    }
    try testing.expect(has_variation);
}

test "pixelateImage - pixelation effect" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a gradient image
    const image = try createGradientImage(allocator, 10, 1);
    ctx.setImage(image);

    // Apply pixelation
    try basic.pixelateImage(&ctx, .{5});

    const pixels = ctx.image.pixels.rgba32;
    // Should have blocky appearance
    // Check that nearby pixels are similar
    const diff = if (pixels[0].r > pixels[4].r) pixels[0].r - pixels[4].r else pixels[4].r - pixels[0].r;
    try testing.expect(diff < 10);
}

test "oilPaintingImage - oil paint effect" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test image
    const image = try createTestImage(allocator, 5, 5, img.color.Rgba32{ .r = 128, .g = 128, .b = 128, .a = 255 });
    ctx.setImage(image);

    // Apply oil painting effect
    try basic.oilPaintingImage(&ctx, .{ 3, 10 });

    // Effect should complete without error
    try testing.expect(ctx.image.width == 5);
    try testing.expect(ctx.image.height == 5);
}

// Transform tests
test "flipImage - horizontal flip" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a test pattern
    const image = try img.Image.create(allocator, 3, 1, .rgba32);
    image.pixels.rgba32[0] = img.color.Rgba32{ .r = 255, .g = 0, .b = 0, .a = 255 }; // Red
    image.pixels.rgba32[1] = img.color.Rgba32{ .r = 0, .g = 255, .b = 0, .a = 255 }; // Green
    image.pixels.rgba32[2] = img.color.Rgba32{ .r = 0, .g = 0, .b = 255, .a = 255 }; // Blue
    ctx.setImage(image);

    // Flip horizontal
    try basic.flipImage(&ctx, .{"horizontal"});

    const pixels = ctx.image.pixels.rgba32;
    // Should be reversed: Blue, Green, Red
    try testing.expectEqual(@as(u8, 0), pixels[0].r); // Blue
    try testing.expectEqual(@as(u8, 0), pixels[0].g);
    try testing.expectEqual(@as(u8, 255), pixels[0].b);

    try testing.expectEqual(@as(u8, 255), pixels[2].r); // Red
    try testing.expectEqual(@as(u8, 0), pixels[2].g);
    try testing.expectEqual(@as(u8, 0), pixels[2].b);
}

test "flipImage - vertical flip" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a vertical pattern
    const image = try img.Image.create(allocator, 1, 3, .rgba32);
    image.pixels.rgba32[0] = img.color.Rgba32{ .r = 255, .g = 0, .b = 0, .a = 255 }; // Red
    image.pixels.rgba32[1] = img.color.Rgba32{ .r = 0, .g = 255, .b = 0, .a = 255 }; // Green
    image.pixels.rgba32[2] = img.color.Rgba32{ .r = 0, .g = 0, .b = 255, .a = 255 }; // Blue
    ctx.setImage(image);

    // Flip vertical
    try basic.flipImage(&ctx, .{"vertical"});

    const pixels = ctx.image.pixels.rgba32;
    // Should be reversed: Blue, Green, Red
    try testing.expectEqual(@as(u8, 0), pixels[0].r); // Blue
    try testing.expectEqual(@as(u8, 0), pixels[0].g);
    try testing.expectEqual(@as(u8, 255), pixels[0].b);

    try testing.expectEqual(@as(u8, 255), pixels[2].r); // Red
    try testing.expectEqual(@as(u8, 0), pixels[2].g);
    try testing.expectEqual(@as(u8, 0), pixels[2].b);
}

test "rotateImage - 90 degrees clockwise" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a 2x2 test image
    const image = try img.Image.create(allocator, 2, 2, .rgba32);
    image.pixels.rgba32[0] = img.color.Rgba32{ .r = 255, .g = 0, .b = 0, .a = 255 }; // Top-left: Red
    image.pixels.rgba32[1] = img.color.Rgba32{ .r = 0, .g = 255, .b = 0, .a = 255 }; // Top-right: Green
    image.pixels.rgba32[2] = img.color.Rgba32{ .r = 0, .g = 0, .b = 255, .a = 255 }; // Bottom-left: Blue
    image.pixels.rgba32[3] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 }; // Bottom-right: White
    ctx.setImage(image);

    // Rotate 90 degrees clockwise
    try basic.rotateImage(&ctx, .{90});

    try testing.expectEqual(@as(usize, 2), ctx.image.width);
    try testing.expectEqual(@as(usize, 2), ctx.image.height);

    const pixels = ctx.image.pixels.rgba32;
    // After 90° clockwise: Blue should be at top-left
    try testing.expectEqual(@as(u8, 0), pixels[0].r);
    try testing.expectEqual(@as(u8, 0), pixels[0].g);
    try testing.expectEqual(@as(u8, 255), pixels[0].b);
}

test "rotateImage - 180 degrees" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a 2x2 test image
    const image = try img.Image.create(allocator, 2, 2, .rgba32);
    image.pixels.rgba32[0] = img.color.Rgba32{ .r = 255, .g = 0, .b = 0, .a = 255 }; // Top-left: Red
    image.pixels.rgba32[1] = img.color.Rgba32{ .r = 0, .g = 255, .b = 0, .a = 255 }; // Top-right: Green
    image.pixels.rgba32[2] = img.color.Rgba32{ .r = 0, .g = 0, .b = 255, .a = 255 }; // Bottom-left: Blue
    image.pixels.rgba32[3] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 }; // Bottom-right: White
    ctx.setImage(image);

    // Rotate 180 degrees
    try basic.rotateImage(&ctx, .{180});

    const pixels = ctx.image.pixels.rgba32;
    // After 180°: White should be at top-left
    try testing.expectEqual(@as(u8, 255), pixels[0].r);
    try testing.expectEqual(@as(u8, 255), pixels[0].g);
    try testing.expectEqual(@as(u8, 255), pixels[0].b);
}

test "rotateImage - 270 degrees clockwise" {
    const allocator = std.testing.allocator;
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Create a 2x2 test image
    const image = try img.Image.create(allocator, 2, 2, .rgba32);
    image.pixels.rgba32[0] = img.color.Rgba32{ .r = 255, .g = 0, .b = 0, .a = 255 }; // Top-left: Red
    image.pixels.rgba32[1] = img.color.Rgba32{ .r = 0, .g = 255, .b = 0, .a = 255 }; // Top-right: Green
    image.pixels.rgba32[2] = img.color.Rgba32{ .r = 0, .g = 0, .b = 255, .a = 255 }; // Bottom-left: Blue
    image.pixels.rgba32[3] = img.color.Rgba32{ .r = 255, .g = 255, .b = 255, .a = 255 }; // Bottom-right: White
    ctx.setImage(image);

    // Rotate 270 degrees clockwise
    try basic.rotateImage(&ctx, .{270});

    const pixels = ctx.image.pixels.rgba32;
    // After 270° clockwise: Green should be at top-left
    try testing.expectEqual(@as(u8, 0), pixels[0].r);
    try testing.expectEqual(@as(u8, 255), pixels[0].g);
    try testing.expectEqual(@as(u8, 0), pixels[0].b);
}
