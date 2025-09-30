const std = @import("std");
const mimg = @import("mimg");
const fs = std.fs;
const img = @import("zigimg");
const debug = std.debug;
const log = std.log;

const LENA = "examples/lena.png";

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var argsIterator = try std.process.ArgIterator.initWithAllocator(allocator);
    defer argsIterator.deinit();

    _ = argsIterator.next(); // Skip executable name

    const filename = argsIterator.next();
    var image: img.Image = undefined;
    defer image.deinit(allocator);
    if (filename) |file_name| {
        // load from zigimg
        image = try loadImage(allocator, file_name);

        std.debug.print("Successfully loaded image from '{s}\n'", .{file_name});
        printImageInfo(&image);
    } else {
        log.info("No command provided. Defaulting to help.\n", .{});
        printHelp();
        return;
    }
    // From here, we can assume we have our input image loaded.

    try modify_image(&image, allocator);

    const output_filename = argsIterator.next() orelse "out.png";
    log.info("Saving image to '{s}'\n", .{output_filename});
    try saveImage(&image, allocator, output_filename);
    log.info("Image saved successfully.\n", .{});
}

pub fn printHelp() void {
    log.warn("Unimplemented", .{});
    return;
}

pub fn loadImage(allocator: std.mem.Allocator, path: []const u8) !img.Image {
    var read_buffer: [8192]u8 = undefined;
    return img.Image.fromFilePath(allocator, path, &read_buffer) catch |err| {
        std.log.err("Failed to load image '{s}': {}", .{ path, err });
        return err;
    };
}

pub fn saveImage(image: *const img.Image, allocator: std.mem.Allocator, path: []const u8) !void {
    var write_buffer: [img.io.DEFAULT_BUFFER_SIZE]u8 = undefined;
    try image.writeToFilePath(allocator, path, write_buffer[0..], .{ .png = .{} });
}

pub fn printImageInfo(image: *const img.Image) void {
    log.info("Image Info:", .{});
    log.info(" - Width: {}", .{image.width});
    log.info(" - Height: {}", .{image.height});
    log.info(" - Total Pixels: {}", .{image.width * image.height});
}

pub fn modify_image(image: *img.Image, allocator: std.mem.Allocator) !void {
    try image.convert(allocator, .rgba32);

    for (image.pixels.rgba32) |*pixel| {
        // Convert RGB to grayscale using the luminance formula: 0.299*R + 0.587*G + 0.114*B
        const r = pixel.r;

        pixel.r = 255 - r;
    }
}
