const std = @import("std");
const types = @import("types.zig");
const utils = @import("utils.zig");
const img = @import("zigimg");
const cli = @import("cli.zig");

fn initApp() !struct { allocator: std.mem.Allocator, args: []const []const u8 } {
    const allocator = std.heap.page_allocator;
    var args_iter = try std.process.ArgIterator.initWithAllocator(allocator);
    defer args_iter.deinit();

    // Skip executable name
    _ = args_iter.next();

    // Collect all remaining arguments
    var args_list = std.ArrayListUnmanaged([]const u8){};
    defer args_list.deinit(allocator);

    while (args_iter.next()) |arg| {
        const duped = try allocator.dupe(u8, arg);
        try args_list.append(allocator, duped);
    }

    return .{ .allocator = allocator, .args = try args_list.toOwnedSlice(allocator) };
}

fn parsePreFilenameOptions(ctx: *types.Context, args: []const []const u8, arg_index: *usize) !?[]const u8 {
    while (arg_index.* < args.len) {
        const arg = args[arg_index.*];
        arg_index.* += 1;

        var consumed = false;
        inline for (cli.registered_options) |option| {
            if (cli.matchesOption(option.names, arg)) {
                consumed = true;

                if (option.option_type == types.ArgType.Modifier) {
                    std.log.err("Expected an image path before modifier '{s}'", .{arg});
                    try cli.printHelp(ctx, .{});
                    return null;
                }

                const parsed = utils.parseArgsFromSlice(option.param_types, args, arg_index) catch |err| {
                    cli.reportParseError(arg, option, err);
                    return null;
                };

                try @call(.auto, option.func, .{ ctx, parsed });

                if (option.func == cli.printHelp or option.func == cli.printModifiers) {
                    return null;
                }

                break;
            }
        }

        if (!consumed) {
            // This is the filename
            return arg;
        }
    }
    return null;
}

fn loadImage(ctx: *types.Context, path: []const u8) !void {
    const image = try utils.loadImageFromSource(ctx, path);
    ctx.setImage(image);
    ctx.output_extension = utils.getExtensionFromSource(path);
    if (ctx.verbose) {
        std.log.info("Successfully loaded image from '{s}'", .{path});
        std.log.info("Image dimensions: {}x{}", .{ ctx.image.width, ctx.image.height });
        try cli.printImageInfo(ctx, .{});
    }
}

fn processModifiers(ctx: *types.Context, args: []const []const u8, arg_index: *usize) !void {
    cli.processArgumentsFromSlice(ctx, args, arg_index) catch |err| switch (err) {
        cli.CliError.UnknownArgument, cli.CliError.InvalidArguments => return,
        else => return err,
    };
}

fn saveImage(ctx: *types.Context) !void {
    var path_buffer: [utils.output_path_buffer_size]u8 = undefined;
    const output_path = try utils.resolveOutputPath(ctx, ctx.output_filename, &path_buffer);
    if (ctx.verbose) {
        std.log.info("Saving image to '{s}'", .{output_path});
    }
    try utils.saveImage(ctx, output_path);
    if (ctx.verbose) {
        std.log.info("Image saved successfully.", .{});
    }
}

pub fn run() !void {
    var app_data = try initApp();
    defer {
        for (app_data.args) |arg| {
            app_data.allocator.free(arg);
        }
        app_data.allocator.free(app_data.args);
    }

    var ctx = types.Context.init(app_data.allocator);
    defer ctx.deinit();

    var arg_index: usize = 0;
    const filename = try parsePreFilenameOptions(&ctx, app_data.args, &arg_index);
    if (filename) |path| {
        try loadImage(&ctx, path);
    } else {
        std.log.err("No image file provided", .{});
        try cli.printHelp(&ctx, .{});
        return;
    }

    try processModifiers(&ctx, app_data.args, &arg_index);

    try saveImage(&ctx);
}
