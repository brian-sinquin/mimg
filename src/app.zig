const std = @import("std");
const types = @import("types.zig");
const utils = @import("utils.zig");
const cli = @import("cli.zig");

pub fn run() !void {
    const allocator = std.heap.page_allocator;
    var args_iter = try std.process.ArgIterator.initWithAllocator(allocator);
    defer args_iter.deinit();

    // Skip executable name
    _ = args_iter.next();

    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    var filename: ?[]const u8 = null;

    while (filename == null) {
        const next_arg = args_iter.next() orelse break;

        var consumed = false;
        inline for (cli.registered_options) |option| {
            if (cli.matchesOption(option.names, next_arg)) {
                consumed = true;

                if (option.option_type == types.ArgType.Modifier) {
                    std.log.err("Expected an image path before modifier '{s}'", .{next_arg});
                    try cli.printHelp(&ctx, .{});
                    return;
                }

                const parsed = utils.parseArgs(option.param_types, &args_iter) catch |err| {
                    cli.reportParseError(next_arg, option, err);
                    return;
                };

                try @call(.auto, option.func, .{ &ctx, parsed });

                if (option.func == cli.printHelp or option.func == cli.printModifiers) {
                    return;
                }

                break;
            }
        }

        if (consumed) {
            continue;
        }

        filename = next_arg;
    }

    if (filename) |path| {
        const image = try utils.loadImage(&ctx, path);
        ctx.setImage(image);
        std.log.info("Successfully loaded image from '{s}'", .{path});
        if (ctx.verbose) {
            std.log.info("Image dimensions: {}x{}", .{ ctx.image.width, ctx.image.height });
        }
        try cli.printImageInfo(&ctx, .{});
    } else {
        std.log.info("No image provided. Displaying help.", .{});
        try cli.printHelp(&ctx, .{});
        return;
    }

    cli.processArguments(&ctx, &args_iter) catch |err| switch (err) {
        cli.CliError.UnknownArgument, cli.CliError.InvalidArguments => return,
        else => return err,
    };

    var path_buffer: [utils.output_path_buffer_size]u8 = undefined;
    const output_path = try utils.resolveOutputPath(&ctx, ctx.output_filename, &path_buffer);
    if (ctx.verbose) {
        std.log.info("Saving image to '{s}'", .{output_path});
    }
    try utils.saveImageToPath(&ctx, output_path);
    std.log.info("Image saved successfully.", .{});
}
