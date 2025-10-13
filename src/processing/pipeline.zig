const std = @import("std");
const types = @import("../core/types.zig");
const utils = @import("../core/utils.zig");
const cli = @import("../core/cli.zig");
const modifiers = @import("modifiers.zig");
const argument_parser = @import("../core/argument_parser.zig");

/// Operation to be performed on the image
pub const Operation = union(enum) {
    load: []const u8,
    apply_preset,
    global_option: struct {
        name: []const u8,
        args: []const []const u8,
    },
    modifier: struct {
        name: []const u8,
        args: []const []const u8,
    },
    save,
};

/// Image processing pipeline that handles the full workflow
pub const Pipeline = struct {
    ctx: *types.Context,
    operations: std.ArrayListUnmanaged(Operation),

    pub fn init(ctx: *types.Context, allocator: std.mem.Allocator) Pipeline {
        return .{
            .ctx = ctx,
            .operations = std.ArrayListUnmanaged(Operation).initCapacity(allocator, 0) catch unreachable,
        };
    }

    pub fn deinit(self: *Pipeline) void {
        self.operations.deinit(self.ctx.allocator);
        for (self.operations.items) |op| {
            switch (op) {
                .global_option => |go| {
                    self.ctx.allocator.free(go.name);
                    self.ctx.allocator.free(go.args);
                },
                .modifier => |mod| {
                    self.ctx.allocator.free(mod.name);
                    self.ctx.allocator.free(mod.args);
                },
                .load => |path| self.ctx.allocator.free(path),
                else => {},
            }
        }
    }

    /// Add load operation
    pub fn addLoad(self: *Pipeline, path: []const u8) !void {
        try self.operations.append(self.ctx.allocator, .{ .load = try self.ctx.allocator.dupe(u8, path) });
    }

    /// Add preset application
    pub fn addPreset(self: *Pipeline) !void {
        try self.operations.append(self.ctx.allocator, .apply_preset);
    }

    /// Add global option
    pub fn addGlobalOption(self: *Pipeline, option: types.Argument, args: []const []const u8) !void {
        const duped_name = try self.ctx.allocator.dupe(u8, cli.getOptionName(option.names));
        const duped_args = try self.ctx.allocator.dupe([]const u8, args);
        try self.operations.append(self.ctx.allocator, .{ .global_option = .{ .name = duped_name, .args = duped_args } });
    }

    /// Add modifier
    pub fn addModifier(self: *Pipeline, modifier: types.Argument, args: []const []const u8) !void {
        const duped_name = try self.ctx.allocator.dupe(u8, cli.getOptionName(modifier.names));
        const duped_args = try self.ctx.allocator.dupe([]const u8, args);
        try self.operations.append(self.ctx.allocator, .{ .modifier = .{ .name = duped_name, .args = duped_args } });
    }

    /// Add save operation
    pub fn addSave(self: *Pipeline) !void {
        try self.operations.append(self.ctx.allocator, .save);
    }

    /// Execute all operations
    pub fn execute(self: *Pipeline) !void {
        for (self.operations.items) |op| {
            switch (op) {
                .load => |path| {
                    const image = try utils.loadImageFromSource(self.ctx, path);
                    self.ctx.setImage(image);
                    self.ctx.output_extension = utils.getExtensionFromSource(path);
                    self.ctx.input_filename = path;
                    if (self.ctx.verbose) {
                        std.log.info("Successfully loaded image from '{s}'", .{path});
                        std.log.info("Image dimensions: {}x{}", .{ self.ctx.image.width, self.ctx.image.height });
                        try cli.printImageInfo(self.ctx, .{});
                    }
                },
                .apply_preset => {
                    try cli.applyPreset(self.ctx);
                },
                .global_option => |go| {
                    const option = for (cli.registered_options) |opt| {
                        if (std.mem.eql(u8, cli.getOptionName(opt.names), go.name)) break opt;
                    } else {
                        std.log.err("Unknown global option: {s}", .{go.name});
                        return error.UnknownOption;
                    };
                    var local_index: usize = 0;
                    const parsed = utils.parseArgsFromSlice(option.param_types, go.args, &local_index) catch |err| {
                        std.log.err("Error parsing args for global option '{s}': {}", .{ go.name, err });
                        return err;
                    };
                    try @call(.auto, option.func, .{ self.ctx, parsed });
                },
                .modifier => |mod| {
                    const modifier = for (cli.registered_options) |opt| {
                        if (std.mem.eql(u8, cli.getOptionName(opt.names), mod.name)) break opt;
                    } else {
                        std.log.err("Unknown modifier: {s}", .{mod.name});
                        return error.UnknownModifier;
                    };
                    var local_index: usize = 0;
                    const parsed = utils.parseArgsFromSlice(modifier.param_types, mod.args, &local_index) catch |err| {
                        std.log.err("Error parsing args for modifier '{s}': {}", .{ mod.name, err });
                        return err;
                    };
                    try @call(.auto, modifier.func, .{ self.ctx, parsed });
                },
                .save => {
                    var path_buffer: [utils.output_path_buffer_size]u8 = undefined;
                    const output_path = try utils.resolveOutputPath(self.ctx, self.ctx.output_filename, &path_buffer);
                    if (self.ctx.verbose) {
                        std.log.info("Saving image to '{s}'", .{output_path});
                    }
                    try utils.saveImage(self.ctx, output_path);
                    if (self.ctx.verbose) {
                        std.log.info("Image saved successfully.", .{});
                    }
                },
            }
        }
    }
};
