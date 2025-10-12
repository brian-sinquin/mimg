const std = @import("std");
const types = @import("types.zig");
const utils = @import("utils.zig");
const modifiers = @import("../processing/modifiers.zig");

// Global options (non-modifiers)
pub const global_options = [_]types.Argument{
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
        .names = .{ .pair = .{ .short = "-p", .long = "--preset" } },
        .option_type = types.ArgType.Option,
        .param_types = &[_]type{[]const u8},
        .description = "Apply modifiers from a preset file (one chain per line)",
        .usage = "--preset <file>",
        .func = applyPresetFile,
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

// Combine all options for processing
pub const all_options = modifiers.modifiers ++ global_options;

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

    const extension = args[0];
    // Input validation
    if (extension.len == 0) return error.InvalidExtension;
    if (extension[0] != '.') return error.InvalidExtension;

    ctx.output_extension = ctx.allocator.dupe(u8, extension) catch unreachable;
}

fn enableVerbose(ctx: *types.Context, args: anytype) !void {
    _ = args;
    ctx.setVerbose(true);
}

fn applyPresetFile(ctx: *types.Context, args: anytype) !void {
    const preset_path = args[0];
    if (ctx.preset_path) |old_path| {
        ctx.allocator.free(old_path);
    }
    ctx.preset_path = ctx.allocator.dupe(u8, preset_path) catch unreachable;
}

pub fn applyPreset(ctx: *types.Context) !void {
    if (ctx.preset_path) |preset_path| {
        // Read the preset file
        const file = try std.fs.cwd().openFile(preset_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const content = try ctx.allocator.alloc(u8, file_size);
        defer ctx.allocator.free(content);

        _ = try file.readAll(content);

        // Parse lines and apply each as a modifier chain
        var line_iter = std.mem.splitSequence(u8, content, "\n");
        while (line_iter.next()) |line| {
            // Skip empty lines and comments
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0 or trimmed[0] == '#') continue;

            // Parse the line as arguments
            var args_buf: [64][]const u8 = undefined;
            var args_count: usize = 0;

            var word_iter = std.mem.tokenizeSequence(u8, trimmed, " \t");
            while (word_iter.next()) |word| {
                if (args_count < args_buf.len) {
                    args_buf[args_count] = word;
                    args_count += 1;
                }
            }

            if (args_count > 0) {
                // Apply this chain of modifiers
                var temp_index: usize = 0;
                try processArgumentsFromSlice(ctx, args_buf[0..args_count], &temp_index);
            }
        }
    }
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
    inline for (modifiers.modifiers) |modifier| {
        const names_str = getOptionName(modifier.names);
        std.debug.print("  {s: <18} {s}\n", .{ names_str, modifier.description });
        if (modifier.usage.len > 0) {
            std.debug.print("    usage: {s}\n", .{modifier.usage});
        }
    }
}

fn printModifiersInCategory(modifier_names: []const []const u8) void {
    inline for (modifiers.modifiers) |modifier| {
        const names_str = getOptionName(modifier.names);
        const is_in_category = blk: {
            for (modifier_names) |mod_name| {
                if (std.mem.eql(u8, names_str, mod_name)) break :blk true;
            }
            break :blk false;
        };

        if (is_in_category) {
            std.debug.print("  {s: <18} {s}\n", .{ names_str, modifier.description });
        }
    }
}

pub fn printVersion(ctx: *types.Context, args: anytype) !void {
    _ = ctx;
    _ = args;
    const version = @import("build_options").version;
    std.debug.print("mimg v{s}\n", .{version});
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
        return types.CliError.UnknownArgument;
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
        return types.CliError.UnknownArgument;
    }
}

fn handleArgument(ctx: *types.Context, iterator: *std.process.ArgIterator, arg: []const u8) !bool {
    inline for (all_options) |option| {
        if (matchesOption(option.names, arg)) {
            const parsed = utils.parseArgs(option.param_types, iterator) catch |parse_err| {
                reportParseError(arg, option, parse_err);
                return types.CliError.InvalidArguments;
            };
            @call(.auto, option.func, .{ ctx, parsed }) catch |modifier_err| {
                std.log.err("Error applying modifier '{s}': {}", .{ arg, modifier_err });
                return types.CliError.InvalidArguments;
            };
            return true;
        }
    }
    return false;
}

fn handleArgumentFromSlice(ctx: *types.Context, args: []const []const u8, arg_index: *usize, arg: []const u8) !bool {
    inline for (all_options) |option| {
        if (matchesOption(option.names, arg)) {
            const parsed = utils.parseArgsFromSlice(option.param_types, args, arg_index) catch |parse_err| {
                reportParseError(arg, option, parse_err);
                return types.CliError.InvalidArguments;
            };
            @call(.auto, option.func, .{ ctx, parsed }) catch |modifier_err| {
                std.log.err("Error applying modifier '{s}': {}", .{ arg, modifier_err });
                return types.CliError.InvalidArguments;
            };

            // Append modifier name and parameters to output filename for modifiers
            if (option.option_type == types.ArgType.Modifier) {
                try appendModifierToOutputFilename(ctx, arg, option.param_types, parsed);
            }

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

    const version = @import("build_options").version;
    std.debug.print("mimg v{s} - Command-line image processing tool\n", .{version});
    std.debug.print("==========================================\n\n", .{});

    std.debug.print("USAGE:\n", .{});
    std.debug.print("  mimg <image_path> [modifiers] [options]\n", .{});
    std.debug.print("  mimg <image_path> --preset <preset_file> [options]\n\n", .{});

    std.debug.print("EXAMPLES:\n", .{});
    std.debug.print("  mimg input.png brightness 20 saturation 1.3 -o output.png\n", .{});
    std.debug.print("  mimg *.jpg grayscale vignette 0.4 -d processed/\n", .{});
    std.debug.print("  mimg photo.png resize 800 600 crop 100 100 600 400 sharpen\n\n", .{});

    // Color Adjustments
    std.debug.print("COLOR ADJUSTMENTS:\n", .{});
    printModifiersInCategory(&[_][]const u8{ "brightness", "contrast", "saturation", "hue-shift", "gamma", "exposure", "vibrance", "equalize" });

    // Color Effects
    std.debug.print("\nCOLOR EFFECTS:\n", .{});
    printModifiersInCategory(&[_][]const u8{ "grayscale", "sepia", "invert", "threshold", "solarize", "posterize", "colorize", "duotone" });

    // Filters & Effects
    std.debug.print("\nFILTERS & EFFECTS:\n", .{});
    printModifiersInCategory(&[_][]const u8{ "blur", "gaussian-blur", "sharpen", "edge-detect", "emboss", "median-filter", "noise", "vignette", "pixelate", "oil-painting" });

    // Geometric Transforms
    std.debug.print("\nGEOMETRIC TRANSFORMS:\n", .{});
    printModifiersInCategory(&[_][]const u8{ "resize", "crop", "rotate", "flip" });

    // Global Options
    std.debug.print("\nGLOBAL OPTIONS:\n", .{});
    inline for (global_options) |option| {
        const names_str = getOptionName(option.names);
        std.debug.print("  {s: <20} {s}\n", .{ names_str, option.description });
    }

    std.debug.print("\nFor detailed modifier usage, run: mimg --list-modifiers\n", .{});
    std.debug.print("For more examples, see: https://github.com/brian-sinquin/mimg\n", .{});
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
    return all_options[0..];
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

fn appendModifierToOutputFilename(ctx: *types.Context, modifier_name: []const u8, param_types: []const type, parsed_args: anytype) !void {
    // Input validation
    if (modifier_name.len == 0) return error.InvalidModifierName;
    const max_filename_length = 255; // Reasonable filename length limit

    // Allocate a new buffer for the updated filename
    var new_filename = std.ArrayList(u8).initCapacity(ctx.allocator, ctx.output_filename.len + modifier_name.len + 32) catch unreachable;
    defer new_filename.deinit(ctx.allocator);

    // Start with current output filename
    if (std.mem.eql(u8, ctx.output_filename, "out")) {
        // First modifier, replace "out" entirely
        new_filename.appendSliceAssumeCapacity(modifier_name);
    } else {
        // Subsequent modifiers, append with underscore
        new_filename.appendSliceAssumeCapacity(ctx.output_filename);
        new_filename.appendAssumeCapacity('_');
        new_filename.appendSliceAssumeCapacity(modifier_name);
    }

    // Add parameters if any
    inline for (param_types, 0..) |param_type, i| {
        if (param_type != []const u8) {
            // Add underscore before parameter
            new_filename.appendAssumeCapacity('_');
            // Convert parameter to string and append
            switch (param_type) {
                u8, u16, u32, usize, i8, i16, i32, isize => {
                    const value = @as(param_type, parsed_args[i]);
                    const str = std.fmt.bufPrint(new_filename.unusedCapacitySlice(), "{}", .{value}) catch unreachable;
                    new_filename.items.len += str.len;
                },
                f32, f64 => {
                    const value = @as(param_type, parsed_args[i]);
                    const str = std.fmt.bufPrint(new_filename.unusedCapacitySlice(), "{d}", .{value}) catch unreachable;
                    new_filename.items.len += str.len;
                },
                else => {
                    // For other types, just skip
                },
            }
        }
    }

    // Check filename length
    if (new_filename.items.len > max_filename_length) {
        return error.FilenameTooLong;
    }

    // Update the context's output filename
    ctx.allocator.free(ctx.output_filename);
    ctx.output_filename = new_filename.toOwnedSlice(ctx.allocator) catch unreachable;
}
