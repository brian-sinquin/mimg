const std = @import("std");
const types = @import("types.zig");
const utils = @import("utils.zig");
const basic = @import("basic.zig");

// Version is set at build time via build options
pub const VERSION: []const u8 = @import("build_options").version;

pub const CliError = error{
    UnknownArgument,
    InvalidArguments,
};

const options = [_]types.Argument{
    .{
        .names = .{ .single = "invert" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{},
        .description = "Invert the colors of the current image",
        .usage = "invert",
        .func = basic.invertColors,
    },
    .{
        .names = .{ .single = "resize" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{ u16, u16 },
        .description = "Resize the image using nearest-neighbor sampling",
        .usage = "resize <width> <height>",
        .func = basic.resizeImage,
    },
    .{
        .names = .{ .single = "crop" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{ u16, u16, u16, u16 },
        .description = "Crop the image using top-left coordinate and size",
        .usage = "crop <x> <y> <width> <height>",
        .func = basic.cropImage,
    },
    .{
        .names = .{ .single = "rotate" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{f64},
        .description = "Rotate the image clockwise by any angle (auto-resizes canvas)",
        .usage = "rotate <degrees>",
        .func = basic.rotateImage,
    },
    .{
        .names = .{ .single = "flip" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{[]const u8},
        .description = "Flip the image horizontally or vertically",
        .usage = "flip <horizontal|vertical>",
        .func = basic.flipImage,
    },
    .{
        .names = .{ .single = "grayscale" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{},
        .description = "Convert the image to grayscale",
        .usage = "grayscale",
        .func = basic.grayscaleImage,
    },
    .{
        .names = .{ .single = "brightness" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{i8},
        .description = "Adjust image brightness",
        .usage = "brightness <value (-128 to 127)>",
        .func = basic.adjustBrightness,
    },
    .{
        .names = .{ .single = "blur" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{u8},
        .description = "Apply a simple box blur",
        .usage = "blur <kernel_size (odd)>",
        .func = basic.blurImage,
    },
    .{
        .names = .{ .single = "saturation" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{f32},
        .description = "Adjust color saturation",
        .usage = "saturation <factor>",
        .func = basic.adjustSaturation,
    },
    .{
        .names = .{ .single = "contrast" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{f32},
        .description = "Adjust image contrast",
        .usage = "contrast <factor>",
        .func = basic.adjustContrast,
    },
    .{
        .names = .{ .single = "gamma" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{f32},
        .description = "Apply gamma correction",
        .usage = "gamma <value>",
        .func = basic.adjustGamma,
    },
    .{
        .names = .{ .single = "sepia" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{},
        .description = "Apply sepia tone effect",
        .usage = "sepia",
        .func = basic.applySepia,
    },
    .{
        .names = .{ .single = "sharpen" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{},
        .description = "Sharpen the image",
        .usage = "sharpen",
        .func = basic.sharpenImage,
    },
    .{
        .names = .{ .single = "gaussian-blur" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{f32},
        .description = "Apply Gaussian blur with configurable sigma",
        .usage = "gaussian-blur <sigma>",
        .func = basic.gaussianBlurImage,
    },
    .{
        .names = .{ .single = "emboss" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{},
        .description = "Apply emboss effect for 3D-like appearance",
        .usage = "emboss",
        .func = basic.embossImage,
    },
    .{
        .names = .{ .single = "vignette" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{f32},
        .description = "Apply vignette effect to darken image corners",
        .usage = "vignette <intensity (0.0-1.0)>",
        .func = basic.vignetteImage,
    },
    .{
        .names = .{ .single = "posterize" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{u8},
        .description = "Reduce color levels for artistic poster effect",
        .usage = "posterize <levels (2-256)>",
        .func = basic.posterizeImage,
    },
    .{
        .names = .{ .single = "hue-shift" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{f32},
        .description = "Shift the hue of all colors in the image",
        .usage = "hue-shift <degrees (-180 to 180)>",
        .func = basic.hueShiftImage,
    },
    .{
        .names = .{ .single = "median-filter" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{u8},
        .description = "Apply median filter for noise reduction",
        .usage = "median-filter <kernel_size (odd)>",
        .func = basic.medianFilterImage,
    },
    .{
        .names = .{ .single = "threshold" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{u8},
        .description = "Convert image to pure black and white based on luminance threshold",
        .usage = "threshold <value (0-255)>",
        .func = basic.thresholdImage,
    },
    .{
        .names = .{ .single = "solarize" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{u8},
        .description = "Invert colors above threshold for artistic effect",
        .usage = "solarize <threshold (0-255)>",
        .func = basic.solarizeImage,
    },
    .{
        .names = .{ .single = "edge-detect" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{},
        .description = "Detect edges using Sobel operator",
        .usage = "edge-detect",
        .func = basic.edgeDetectImage,
    },
    .{
        .names = .{ .single = "pixelate" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{u8},
        .description = "Apply pixelation/mosaic effect",
        .usage = "pixelate <block_size>",
        .func = basic.pixelateImage,
    },
    .{
        .names = .{ .single = "noise" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{f32},
        .description = "Add random noise to image",
        .usage = "noise <amount (0.0-1.0)>",
        .func = basic.addNoiseImage,
    },
    .{
        .names = .{ .single = "exposure" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{f32},
        .description = "Adjust exposure (like camera EV)",
        .usage = "exposure <value (-2.0 to 2.0)>",
        .func = basic.adjustExposure,
    },
    .{
        .names = .{ .single = "vibrance" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{f32},
        .description = "Adjust vibrance (smart saturation)",
        .usage = "vibrance <factor>",
        .func = basic.adjustVibrance,
    },
    .{
        .names = .{ .single = "equalize" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{},
        .description = "Apply histogram equalization for better contrast",
        .usage = "equalize",
        .func = basic.equalizeImage,
    },
    .{
        .names = .{ .single = "colorize" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{ u8, u8, u8, f32 },
        .description = "Colorize/tint image with RGB color",
        .usage = "colorize <r> <g> <b> <intensity (0.0-1.0)>",
        .func = basic.colorizeImage,
    },
    .{
        .names = .{ .single = "duotone" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{ u8, u8, u8, u8, u8, u8 },
        .description = "Apply duotone effect (Spotify-style)",
        .usage = "duotone <dark_r> <dark_g> <dark_b> <light_r> <light_g> <light_b>",
        .func = basic.duotoneImage,
    },
    .{
        .names = .{ .single = "oil-painting" },
        .option_type = types.ArgType.Modifier,
        .param_types = &[_]type{usize},
        .description = "Apply oil painting artistic effect",
        .usage = "oil-painting <radius>",
        .func = basic.oilPaintingImage,
    },
    .{
        .names = .{ .pair = .{ .short = "-o", .long = "--output" } },
        .option_type = types.ArgType.Option,
        .param_types = &[_]type{[]const u8},
        .description = "Set the output file name for the processed image",
        .usage = "--output <filename>",
        .func = setOutputFilename,
    },
    .{
        .names = .{ .pair = .{ .short = "-d", .long = "--output-dir" } },
        .option_type = types.ArgType.Option,
        .param_types = &[_]type{[]const u8},
        .description = "Set the output directory for the processed image",
        .usage = "--output-dir <directory>",
        .func = setOutputDirectory,
    },
    .{
        .names = .{ .pair = .{ .short = "-e", .long = "--output-extension" } },
        .option_type = types.ArgType.Option,
        .param_types = &[_]type{[]const u8},
        .description = "Set the output file extension (e.g., .png, .jpg)",
        .usage = "--output-extension <extension>",
        .func = setOutputExtension,
    },
    .{
        .names = .{ .pair = .{ .short = "-l", .long = "--list-modifiers" } },
        .option_type = types.ArgType.Option,
        .param_types = &[_]type{},
        .description = "List available modifiers with their usage",
        .usage = "--list-modifiers",
        .func = printModifiers,
    },
    .{
        .names = .{ .pair = .{ .short = "-v", .long = "--verbose" } },
        .option_type = types.ArgType.Option,
        .param_types = &[_]type{},
        .description = "Enable verbose logging",
        .usage = "--verbose",
        .func = enableVerbose,
    },
    .{
        .names = .{ .pair = .{ .short = "-h", .long = "--help" } },
        .option_type = types.ArgType.Option,
        .param_types = &[_]type{},
        .description = "Display command usage information",
        .usage = "--help",
        .func = printHelp,
    },
    .{
        .names = .{ .pair = .{ .short = "-V", .long = "--version" } },
        .option_type = types.ArgType.Option,
        .param_types = &[_]type{},
        .description = "Display version information",
        .usage = "--version",
        .func = printVersion,
    },
};

pub const registered_options = options;

fn setOutputFilename(ctx: *types.Context, args: anytype) !void {
    _ = ctx.image;
    _ = ctx.allocator;
    ctx.setOutput(args[0]);
}

fn setOutputDirectory(ctx: *types.Context, args: anytype) !void {
    _ = ctx.image;
    _ = ctx.allocator;
    ctx.setOutputDirectory(args[0]);
}

fn setOutputExtension(ctx: *types.Context, args: anytype) !void {
    _ = ctx.image;
    _ = ctx.allocator;
    ctx.output_extension = args[0];
}

fn enableVerbose(ctx: *types.Context, args: anytype) !void {
    _ = args;
    ctx.setVerbose(true);
}

pub fn getOptionName(names: types.Names) []const u8 {
    return switch (names) {
        .single => |name| name,
        .pair => |pair| if (pair.long.len > 0) pair.long else pair.short,
    };
}

pub fn printModifiers(ctx: *types.Context, args: anytype) !void {
    _ = ctx;
    _ = args;
    std.debug.print("Available modifiers:\n", .{});
    inline for (registered_options) |option| {
        if (option.option_type != types.ArgType.Modifier) continue;
        const names_str = getOptionName(option.names);
        std.debug.print("  {s: <18} {s}\n", .{ names_str, option.description });
        if (option.usage.len > 0) {
            std.debug.print("    usage: {s}\n", .{option.usage});
        }
    }
}

pub fn printVersion(ctx: *types.Context, args: anytype) !void {
    _ = ctx;
    _ = args;
    std.debug.print("mimg v{s}\n", .{VERSION});
    std.debug.print("A command-line image manipulation tool built with Zig\n", .{});
    std.process.exit(0);
}

pub fn processArguments(ctx: *types.Context, iterator: *std.process.ArgIterator) !void {
    while (iterator.next()) |arg| {
        if (try handleArgument(ctx, iterator, arg)) {
            continue;
        }
        std.log.warn("Unknown argument: '{s}'", .{arg});
        try printHelp(ctx, .{});
        return CliError.UnknownArgument;
    }
}

pub fn processArgumentsFromSlice(ctx: *types.Context, args: []const []const u8, arg_index: *usize) !void {
    while (arg_index.* < args.len) {
        const arg = args[arg_index.*];
        arg_index.* += 1;

        if (try handleArgumentFromSlice(ctx, args, arg_index, arg)) {
            continue;
        }
        std.log.warn("Unknown argument: '{s}'", .{arg});
        try printHelp(ctx, .{});
        return CliError.UnknownArgument;
    }
}

fn handleArgument(ctx: *types.Context, iterator: *std.process.ArgIterator, arg: []const u8) !bool {
    inline for (options) |option| {
        if (matchesOption(option.names, arg)) {
            const parsed = utils.parseArgs(option.param_types, iterator) catch |parse_err| {
                reportParseError(arg, option, parse_err);
                return CliError.InvalidArguments;
            };
            @call(.auto, option.func, .{ ctx, parsed }) catch |modifier_err| {
                std.log.err("Error applying modifier '{s}': {}", .{ arg, modifier_err });
                return CliError.InvalidArguments;
            };
            return true;
        }
    }
    return false;
}

fn handleArgumentFromSlice(ctx: *types.Context, args: []const []const u8, arg_index: *usize, arg: []const u8) !bool {
    inline for (options) |option| {
        if (matchesOption(option.names, arg)) {
            // Back up the index since we already consumed the option name
            arg_index.* -= 1;
            const parsed = utils.parseArgsFromSlice(option.param_types, args, arg_index) catch |parse_err| {
                reportParseError(arg, option, parse_err);
                return CliError.InvalidArguments;
            };
            // Advance past the option name
            arg_index.* += 1;
            @call(.auto, option.func, .{ ctx, parsed }) catch |modifier_err| {
                std.log.err("Error applying modifier '{s}': {}", .{ arg, modifier_err });
                return CliError.InvalidArguments;
            };
            return true;
        }
    }
    return false;
}

pub fn matchesOption(names: types.Names, arg: []const u8) bool {
    return switch (names) {
        .single => |name| std.mem.eql(u8, arg, name),
        .pair => |pair| (pair.short.len > 0 and std.mem.eql(u8, arg, pair.short)) or
            (pair.long.len > 0 and std.mem.eql(u8, arg, pair.long)),
    };
}

pub fn printHelp(ctx: *types.Context, args: anytype) !void {
    _ = ctx;
    _ = args;
    std.debug.print("Usage: mimg <image_path> [options]\n", .{});
    std.debug.print("\nOptions:\n", .{});
    inline for (options) |option| {
        const names_str = getOptionName(option.names);
        std.debug.print("  {s: <18} {s}\n", .{ names_str, option.description });
        if (option.usage.len > 0) {
            std.debug.print("    usage: {s}\n", .{option.usage});
        }
    }
}

pub fn printImageInfo(ctx: *types.Context, args: anytype) !void {
    _ = args;
    if (!ctx.verbose) return;
    std.log.info("Image Info:", .{});
    std.log.info(" - Width: {}", .{ctx.image.width});
    std.log.info(" - Height: {}", .{ctx.image.height});
    std.log.info(" - Total Pixels: {}", .{ctx.image.width * ctx.image.height});
}

pub fn getOptions() []const types.Option {
    return options[0..];
}

pub fn reportParseError(arg: []const u8, option: types.Argument, err: types.ParseArgError) void {
    const usage = if (option.usage.len > 0) option.usage else option.description;
    switch (err) {
        types.ParseArgError.MissingArgument => {
            std.log.err(
                "Option '{s}' expects more arguments. Usage: {s}",
                .{ arg, usage },
            );
        },
        types.ParseArgError.InvalidArgument => {
            std.log.err(
                "Invalid argument for '{s}'. Usage: {s}",
                .{ arg, usage },
            );
        },
    }
}
