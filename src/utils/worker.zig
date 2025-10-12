const std = @import("std");
const types = @import("../core/types.zig");
const utils = @import("../core/utils.zig");
const img = @import("zigimg");
const cli = @import("../core/cli.zig");

/// Result of processing a single file
pub const FileProcessResult = enum {
    processed,
    skipped_unreadable,
    skipped_modifier_error,
    failed_save,
};

/// Work item for a worker thread
pub const WorkItem = struct {
    filename: []const u8,
    args: []const []const u8,
    start_arg_index: usize,
    verbose: bool,
    preset_path: ?[]const u8,
    is_batch: bool,
};

/// Result of processing a work item
pub const WorkResult = struct {
    result: FileProcessResult,
    filename: []const u8,
};

/// Process a single file with the given context and arguments
pub fn processFile(ctx: *types.Context, filename: []const u8, args: []const []const u8, start_arg_index: usize) FileProcessResult {
    // Reset output filename for each file
    ctx.allocator.free(ctx.output_filename);
    ctx.output_filename = ctx.allocator.dupe(u8, "out") catch unreachable;

    ctx.input_filename = filename;

    // Try to load the image, but don't fail if it can't be loaded
    loadImage(ctx, filename) catch |err| {
        std.log.warn("Skipping unreadable file '{s}': {}", .{ filename, err });
        return .skipped_unreadable;
    };

    // Apply preset if specified
    cli.applyPreset(ctx) catch |err| {
        std.log.warn("Skipping file '{s}' due to preset error: {}", .{ filename, err });
        return .skipped_modifier_error;
    };

    var arg_index = start_arg_index;
    processModifiers(ctx, args, &arg_index) catch |err| {
        std.log.warn("Skipping file '{s}' due to modifier error: {}", .{ filename, err });
        return .skipped_modifier_error;
    };

    saveImage(ctx) catch |err| {
        std.log.warn("Failed to save processed image for '{s}': {}", .{ filename, err });
        return .failed_save;
    };

    if (ctx.is_batch) {
        std.log.info("Processed '{s}'", .{filename});
    }

    return .processed;
}

/// Load an image from a file path
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

/// Process modifiers and options from command line arguments
fn processModifiers(ctx: *types.Context, args: []const []const u8, arg_index: *usize) !void {
    // First pass: process global options
    var temp_index = arg_index.*;
    while (temp_index < args.len) {
        const arg = args[temp_index];
        temp_index += 1;

        // Check if this is a global option
        inline for (cli.registered_options) |option| {
            if (option.option_type == types.ArgType.Option and cli.matchesOption(option.names, arg)) {
                const parsed = utils.parseArgsFromSlice(option.param_types, args, &temp_index) catch |parse_err| {
                    cli.reportParseError(arg, option, parse_err);
                    return types.CliError.InvalidArguments;
                };
                try @call(.auto, option.func, .{ ctx, parsed });
                break;
            }
        }
    }

    // Second pass: process modifiers
    cli.processArgumentsFromSlice(ctx, args, arg_index) catch |err| switch (err) {
        types.CliError.UnknownArgument, types.CliError.InvalidArguments => return,
        else => return err,
    };
}

/// Save the processed image to disk
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

/// Process a single work item in a worker thread
fn workerProcessFile(allocator: std.mem.Allocator, work_item: WorkItem) !WorkResult {
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    // Copy flags from work item to context
    ctx.verbose = work_item.verbose;
    ctx.is_batch = work_item.is_batch;
    if (work_item.preset_path) |path| {
        ctx.preset_path = allocator.dupe(u8, path) catch unreachable;
    }

    const result = processFile(&ctx, work_item.filename, work_item.args, work_item.start_arg_index);
    return WorkResult{
        .result = result,
        .filename = work_item.filename,
    };
}

/// Worker thread function that processes a batch of work items
fn workerThread(allocator: std.mem.Allocator, work_items: std.ArrayList(WorkItem), results: []WorkResult, start_idx: usize) void {
    var local_idx = start_idx;
    for (work_items.items) |work_item| {
        const result = workerProcessFile(allocator, work_item) catch |err| {
            std.log.err("Worker thread error processing '{}': {}", .{ work_item.filename, err });
            results[local_idx] = WorkResult{ .result = .failed_save, .filename = work_item.filename };
            local_idx += 1;
            continue;
        };
        results[local_idx] = result;
        local_idx += 1;
    }
    var mutable_work_items = work_items;
    mutable_work_items.deinit(allocator);
}

/// Process multiple files using multithreading
pub fn processFilesMultithreaded(
    allocator: std.mem.Allocator,
    filenames: []const []const u8,
    args: []const []const u8,
    start_arg_index: usize,
    verbose: bool,
    preset_path: ?[]const u8,
) !struct {
    processed: usize,
    skipped_unreadable: usize,
    skipped_modifier_error: usize,
    failed_save: usize,
} {
    const num_threads = std.Thread.getCpuCount() catch 4; // fallback to 4 threads
    const actual_threads = @min(num_threads, filenames.len);
    const is_batch = filenames.len > 1;

    // Divide work among threads
    const files_per_thread = filenames.len / actual_threads;
    const extra_files = filenames.len % actual_threads;

    var threads = std.ArrayList(std.Thread).initCapacity(allocator, actual_threads) catch unreachable;
    defer threads.deinit(allocator);

    var results = std.ArrayList(WorkResult).initCapacity(allocator, filenames.len) catch unreachable;
    defer results.deinit(allocator);
    results.expandToCapacity();

    // Spawn threads
    var start_idx: usize = 0;
    var result_idx: usize = 0;
    for (0..actual_threads) |thread_idx| {
        const thread_files = files_per_thread + if (thread_idx < extra_files) @as(usize, 1) else 0;
        const end_idx = start_idx + thread_files;

        // Create work items for this thread
        var thread_work = std.ArrayList(WorkItem).initCapacity(allocator, thread_files) catch unreachable;
        for (start_idx..end_idx) |i| {
            thread_work.appendAssumeCapacity(WorkItem{
                .filename = filenames[i],
                .args = args,
                .start_arg_index = start_arg_index,
                .verbose = verbose,
                .preset_path = preset_path,
                .is_batch = is_batch,
            });
        }

        // Spawn thread with its work
        const thread = std.Thread.spawn(.{}, workerThread, .{ allocator, thread_work, results.items, result_idx }) catch |err| {
            std.log.err("Failed to spawn thread {}: {}", .{ thread_idx, err });
            thread_work.deinit(allocator);
            start_idx = end_idx;
            result_idx += thread_files;
            continue;
        };
        threads.appendAssumeCapacity(thread);

        start_idx = end_idx;
        result_idx += thread_files;
    }

    // Wait for all threads to complete
    for (threads.items) |thread| {
        thread.join();
    }

    // Count results
    var processed_count: usize = 0;
    var skipped_unreadable_count: usize = 0;
    var skipped_modifier_count: usize = 0;
    var failed_save_count: usize = 0;

    for (results.items) |result| {
        switch (result.result) {
            .processed => processed_count += 1,
            .skipped_unreadable => skipped_unreadable_count += 1,
            .skipped_modifier_error => skipped_modifier_count += 1,
            .failed_save => failed_save_count += 1,
        }
    }

    return .{
        .processed = processed_count,
        .skipped_unreadable = skipped_unreadable_count,
        .skipped_modifier_error = skipped_modifier_count,
        .failed_save = failed_save_count,
    };
}

/// Process a single file (non-multithreaded)
pub fn processSingleFile(
    allocator: std.mem.Allocator,
    filename: []const u8,
    args: []const []const u8,
    start_arg_index: usize,
    verbose: bool,
    preset_path: ?[]const u8,
) !FileProcessResult {
    var ctx = types.Context.init(allocator);
    defer ctx.deinit();

    ctx.verbose = verbose;
    ctx.is_batch = false;
    if (preset_path) |path| {
        ctx.preset_path = allocator.dupe(u8, path) catch unreachable;
    }

    return processFile(&ctx, filename, args, start_arg_index);
}
