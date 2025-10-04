const std = @import("std");
const types = @import("types.zig");
const utils = @import("utils.zig");
const basic = @import("basic.zig");

pub const CliError = error{
    UnknownArgument,
    InvalidArguments,
};

const options = [_]types.Option{
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
        .names = .{ .pair = .{ .short = "-L", .long = "--list-modifiers" } },
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

fn enableVerbose(ctx: *types.Context, args: anytype) !void {
    _ = args;
    ctx.setVerbose(true);
}

pub fn printModifiers(ctx: *types.Context, args: anytype) !void {
    _ = ctx;
    _ = args;
    std.debug.print("Available modifiers:\n", .{});
    inline for (registered_options) |option| {
        if (option.option_type != types.ArgType.Modifier) continue;
        const names_str = switch (option.names) {
            .single => |name| name,
            .pair => |pair| if (pair.long.len > 0) pair.long else pair.short,
        };
        std.debug.print("  {s: <18} {s}\n", .{ names_str, option.description });
        if (option.usage.len > 0) {
            std.debug.print("    usage: {s}\n", .{option.usage});
        }
    }
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

fn handleArgument(ctx: *types.Context, iterator: *std.process.ArgIterator, arg: []const u8) !bool {
    inline for (options) |option| {
        if (matchesOption(option.names, arg)) {
            const parsed = utils.parseArgs(option.param_types, iterator) catch |parse_err| {
                reportParseError(arg, option, parse_err);
                return CliError.InvalidArguments;
            };
            try @call(.auto, option.func, .{ ctx, parsed });
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
        const names_str = switch (option.names) {
            .single => |name| name,
            .pair => |pair| if (pair.long.len > 0) pair.long else pair.short,
        };
        std.debug.print("  {s: <18} {s}\n", .{ names_str, option.description });
        if (option.usage.len > 0) {
            std.debug.print("    usage: {s}\n", .{option.usage});
        }
    }
}

pub fn printImageInfo(ctx: *types.Context, args: anytype) !void {
    _ = args;
    std.log.info("Image Info:", .{});
    std.log.info(" - Width: {}", .{ctx.image.width});
    std.log.info(" - Height: {}", .{ctx.image.height});
    std.log.info(" - Total Pixels: {}", .{ctx.image.width * ctx.image.height});
}

pub fn getOptions() []const types.Option {
    return options[0..];
}

pub fn reportParseError(arg: []const u8, option: types.Option, err: types.ParseArgError) void {
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
