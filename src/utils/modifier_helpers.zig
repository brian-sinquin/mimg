const std = @import("std");
const img = @import("zigimg");
const Context = @import("../core/types.zig").Context;
const utils = @import("../core/utils.zig");

/// Wrapper that handles common modifier boilerplate: convertToRgba32 + logVerbose + processing
/// Reduces 3 lines per modifier to 1 line
pub fn processModifier(
    ctx: *Context,
    comptime operation_name: []const u8,
    comptime processFn: anytype,
    args: anytype,
) !void {
    try utils.convertToRgba32(ctx);
    
    // Build verbose message based on args
    if (@typeInfo(@TypeOf(args)).Struct.fields.len > 0) {
        utils.logVerbose(ctx, "Applying " ++ operation_name ++ " with params {any}", .{args});
    } else {
        utils.logVerbose(ctx, "Applying " ++ operation_name, .{});
    }
    
    try processFn(ctx, args);
}

/// Apply the same function to R, G, B channels of a pixel
/// Preserves alpha channel unchanged
pub inline fn applyToRGB(
    pixel: img.color.Rgba32,
    comptime func: fn (u8) u8,
) img.color.Rgba32 {
    return img.color.Rgba32{
        .r = func(pixel.r),
        .g = func(pixel.g),
        .b = func(pixel.b),
        .a = pixel.a,
    };
}

/// Apply the same float function to R, G, B channels of a pixel
pub inline fn applyToRGBFloat(
    pixel: img.color.Rgba32,
    comptime func: fn (f32) f32,
) img.color.Rgba32 {
    return img.color.Rgba32{
        .r = utils.clampU8(func(@as(f32, @floatFromInt(pixel.r)))),
        .g = utils.clampU8(func(@as(f32, @floatFromInt(pixel.g)))),
        .b = utils.clampU8(func(@as(f32, @floatFromInt(pixel.b)))),
        .a = pixel.a,
    };
}

/// Apply a function with context to R, G, B channels
pub inline fn applyToRGBWithContext(
    pixel: img.color.Rgba32,
    context: anytype,
    comptime func: fn (u8, @TypeOf(context)) u8,
) img.color.Rgba32 {
    return img.color.Rgba32{
        .r = func(pixel.r, context),
        .g = func(pixel.g, context),
        .b = func(pixel.b, context),
        .a = pixel.a,
    };
}

/// Apply a function with context to R, G, B channels (float version)
pub inline fn applyToRGBFloatWithContext(
    pixel: img.color.Rgba32,
    context: anytype,
    comptime func: fn (f32, @TypeOf(context)) f32,
) img.color.Rgba32 {
    return img.color.Rgba32{
        .r = utils.clampU8(func(@as(f32, @floatFromInt(pixel.r)), context)),
        .g = utils.clampU8(func(@as(f32, @floatFromInt(pixel.g)), context)),
        .b = utils.clampU8(func(@as(f32, @floatFromInt(pixel.b)), context)),
        .a = pixel.a,
    };
}

/// Generic matrix transformation for RGB (used in sepia, colorize, etc.)
pub inline fn applyMatrix3x3(
    pixel: img.color.Rgba32,
    matrix: [3][3]f32,
) img.color.Rgba32 {
    const r = @as(f32, @floatFromInt(pixel.r));
    const g = @as(f32, @floatFromInt(pixel.g));
    const b = @as(f32, @floatFromInt(pixel.b));
    
    return img.color.Rgba32{
        .r = utils.clampU8(matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b),
        .g = utils.clampU8(matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b),
        .b = utils.clampU8(matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b),
        .a = pixel.a,
    };
}
