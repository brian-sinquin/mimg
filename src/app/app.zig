const std = @import("std");
const types = @import("../core/types.zig");
const utils = @import("../core/utils.zig");
const cli = @import("../core/cli.zig");
const worker = @import("../utils/worker.zig");

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

fn parsePreFilenameOptions(ctx: *types.Context, args: []const []const u8, arg_index: *usize) !?[]const []const u8 {
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
            // This is the filename pattern
            const filenames = try utils.expandWildcard(ctx.allocator, arg);
            return filenames;
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
    const filenames = try parsePreFilenameOptions(&ctx, app_data.args, &arg_index);
    if (filenames) |files| {
        defer app_data.allocator.free(files);
        if (files.len == 0) {
            std.log.err("No files found matching the pattern", .{});
            return;
        }

        if (files.len == 1) {
            // Single file processing
            const result = try worker.processSingleFile(
                app_data.allocator,
                files[0],
                app_data.args,
                arg_index,
                ctx.verbose,
                ctx.preset_path,
            );

            switch (result) {
                .processed => {},
                .skipped_unreadable => std.log.warn("File was unreadable", .{}),
                .skipped_modifier_error => std.log.warn("Modifier error occurred", .{}),
                .failed_save => std.log.err("Failed to save image", .{}),
            }
        } else {
            // Batch processing with multithreading
            const stats = try worker.processFilesMultithreaded(
                app_data.allocator,
                files,
                app_data.args,
                arg_index,
                ctx.verbose,
                ctx.preset_path,
            );

            // Print summary
            std.log.info("Processing complete:", .{});
            std.log.info("  Files processed successfully: {}", .{stats.processed});
            if (stats.skipped_unreadable > 0) {
                std.log.info("  Files skipped (unreadable): {}", .{stats.skipped_unreadable});
            }
            if (stats.skipped_modifier_error > 0) {
                std.log.info("  Files skipped (modifier error): {}", .{stats.skipped_modifier_error});
            }
            if (stats.failed_save > 0) {
                std.log.info("  Files failed to save: {}", .{stats.failed_save});
            }
        }
    } else {
        std.log.err("No image file provided", .{});
        try cli.printHelp(&ctx, .{});
        return;
    }
}
