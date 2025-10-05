const img = @import("zigimg");
const std = @import("std");

pub const ParseArgError = error{
    MissingArgument,
    InvalidArgument,
};

pub const ArgType = enum {
    Option,
    Modifier,
};

pub const Context = struct {
    image: img.Image,
    allocator: std.mem.Allocator,
    output_filename: []const u8,
    output_directory: ?[]const u8 = null,
    image_loaded: bool = false,
    verbose: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{
            .image = undefined,
            .allocator = allocator,
            .output_filename = "out.png",
            .output_directory = null,
            .image_loaded = false,
            .verbose = false,
        };
    }

    pub fn setImage(self: *Context, image: img.Image) void {
        if (self.image_loaded) {
            self.image.deinit(self.allocator);
        }
        self.image = image;
        self.image_loaded = true;
    }

    pub fn deinit(self: *Context) void {
        if (self.image_loaded) {
            self.image.deinit(self.allocator);
            self.image_loaded = false;
        }
    }

    pub fn setOutput(self: *Context, filename: []const u8) void {
        self.output_filename = filename;
    }

    pub fn setOutputDirectory(self: *Context, directory: []const u8) void {
        self.output_directory = directory;
    }

    pub fn setVerbose(self: *Context, value: bool) void {
        self.verbose = value;
    }
};

pub const Names = union(enum) {
    single: []const u8,
    pair: struct { short: []const u8, long: []const u8 },
};

pub const Argument = struct {
    names: Names,
    option_type: ArgType,
    param_types: []const type,
    description: []const u8,
    usage: []const u8 = "",
    func: *const fn (*Context, anytype) anyerror!void,
};
