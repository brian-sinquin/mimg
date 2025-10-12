const img = @import("zigimg");
const std = @import("std");
const Context = @import("types.zig").Context;
const utils = @import("utils.zig");
const color = @import("color.zig");
const filters = @import("filters.zig");
const transforms = @import("transforms.zig");
const math = std.math;

fn createTempBuffer(ctx: *Context, pixels: []img.color.Rgba32) ![]img.color.Rgba32 {
    return ctx.copyToTempBuffer(pixels);
}

fn clampPixel(value: anytype) @TypeOf(value) {
    return std.math.clamp(value, 0, 255);
}

// Color adjustment functions - delegate to color module
pub fn invertColors(ctx: *Context, args: anytype) !void {
    try color.invertColors(ctx, args);
}

pub fn grayscaleImage(ctx: *Context, args: anytype) !void {
    try color.grayscaleImage(ctx, args);
}

pub fn adjustBrightness(ctx: *Context, args: anytype) !void {
    try color.adjustBrightness(ctx, args);
}

pub fn adjustSaturation(ctx: *Context, args: anytype) !void {
    try color.adjustSaturation(ctx, args);
}

pub fn adjustContrast(ctx: *Context, args: anytype) !void {
    try color.adjustContrast(ctx, args);
}

pub fn adjustGamma(ctx: *Context, args: anytype) !void {
    try color.adjustGamma(ctx, args);
}

pub fn adjustExposure(ctx: *Context, args: anytype) !void {
    try color.adjustExposure(ctx, args);
}

pub fn adjustVibrance(ctx: *Context, args: anytype) !void {
    try color.adjustVibrance(ctx, args);
}

pub fn equalizeImage(ctx: *Context, args: anytype) !void {
    try color.equalizeImage(ctx, args);
}

pub fn hueShiftImage(ctx: *Context, args: anytype) !void {
    try color.hueShiftImage(ctx, args);
}

pub fn applySepia(ctx: *Context, args: anytype) !void {
    try color.applySepia(ctx, args);
}

pub fn colorizeImage(ctx: *Context, args: anytype) !void {
    try color.colorizeImage(ctx, args);
}

pub fn duotoneImage(ctx: *Context, args: anytype) !void {
    try color.duotoneImage(ctx, args);
}

pub fn thresholdImage(ctx: *Context, args: anytype) !void {
    try color.thresholdImage(ctx, args);
}

pub fn solarizeImage(ctx: *Context, args: anytype) !void {
    try color.solarizeImage(ctx, args);
}

pub fn posterizeImage(ctx: *Context, args: anytype) !void {
    try color.posterizeImage(ctx, args);
}

// Filter functions - delegate to filters module
pub fn blurImage(ctx: *Context, args: anytype) !void {
    try filters.blurImage(ctx, args);
}

pub fn gaussianBlurImage(ctx: *Context, args: anytype) !void {
    try filters.gaussianBlurImage(ctx, args);
}

pub fn sharpenImage(ctx: *Context, args: anytype) !void {
    try filters.sharpenImage(ctx, args);
}

pub fn embossImage(ctx: *Context, args: anytype) !void {
    try filters.embossImage(ctx, args);
}

pub fn vignetteImage(ctx: *Context, args: anytype) !void {
    try filters.vignetteImage(ctx, args);
}

pub fn edgeDetectImage(ctx: *Context, args: anytype) !void {
    try filters.edgeDetectImage(ctx, args);
}

pub fn medianFilterImage(ctx: *Context, args: anytype) !void {
    try filters.medianFilterImage(ctx, args);
}

pub fn addNoiseImage(ctx: *Context, args: anytype) !void {
    try filters.addNoiseImage(ctx, args);
}

pub fn pixelateImage(ctx: *Context, args: anytype) !void {
    try filters.pixelateImage(ctx, args);
}

pub fn oilPaintingImage(ctx: *Context, args: anytype) !void {
    try filters.oilPaintingImage(ctx, args);
}

// Transform functions - delegate to transforms module
pub fn resizeImage(ctx: *Context, args: anytype) !void {
    try transforms.resizeImage(ctx, args);
}

pub fn cropImage(ctx: *Context, args: anytype) !void {
    try transforms.cropImage(ctx, args);
}

pub fn flipImage(ctx: *Context, args: anytype) !void {
    const direction = args[0];
    if (std.mem.eql(u8, direction, "horizontal")) {
        try transforms.flipHorizontalImage(ctx, .{});
    } else if (std.mem.eql(u8, direction, "vertical")) {
        try transforms.flipVerticalImage(ctx, .{});
    } else {
        return error.InvalidParameters;
    }
}

pub fn rotateImage(ctx: *Context, args: anytype) !void {
    try transforms.rotateImage(ctx, args);
}
