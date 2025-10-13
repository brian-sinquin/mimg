const img = @import("zigimg");
const std = @import("std");

pub const ParseArgError = error{
    MissingArgument,
    InvalidArgument,
};

pub const CliError = error{
    UnknownArgument,
    InvalidArguments,
};

pub const ImageError = error{
    FileNotFound,
    InvalidFormat,
    UnsupportedFormat,
    CorruptedData,
    OutOfMemory,
    InvalidDimensions,
    LoadFailed,
    SaveFailed,
};

pub const ProcessingError = error{
    InvalidParameters,
    UnsupportedOperation,
    OutOfBounds,
    InsufficientData,
    KernelSizeError,
    UnsupportedRotationAngle,
    InvalidGamma,
    InvalidIntensity,
    InvalidLevels,
    InvalidSigma,
    InvalidNoiseAmount,
    InvalidBlockSize,
    InvalidRadius,
};

pub const FileSystemError = error{
    PathTooLong,
    PermissionDenied,
    DirectoryNotFound,
    FileExists,
    DiskFull,
    AccessDenied,
    InvalidDirectory,
    InvalidFilename,
    InvalidInputFilename,
};

pub const NetworkError = error{
    ConnectionFailed,
    Timeout,
    InvalidUrl,
    HttpError,
    DownloadFailed,
    UrlTooLong,
};

pub const ArgType = enum {
    Option,
    Modifier,
};

pub const Context = struct {
    image: img.Image,
    allocator: std.mem.Allocator,
    output_filename: []const u8,
    input_filename: ?[]const u8,
    output_directory: ?[]const u8 = null,
    output_extension: []const u8 = ".png",
    image_loaded: bool = false,
    verbose: bool = false,
    temp_buffer: ?[]img.color.Rgba32 = null, // Reusable buffer for image processing
    temp_buffer_size: usize = 0, // Track allocated size for reuse
    preset_path: ?[]const u8 = null, // Path to preset file
    is_batch: bool = false, // Whether processing multiple files
    image_cache: std.StringHashMap(img.Image) = undefined, // Cache for loaded images

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{
            .image = undefined,
            .allocator = allocator,
            .output_filename = allocator.dupe(u8, "out") catch unreachable,
            .input_filename = null,
            .output_directory = null,
            .output_extension = ".png",
            .image_loaded = false,
            .verbose = false,
            .temp_buffer = null,
            .temp_buffer_size = 0,
            .preset_path = null,
            .is_batch = false,
            .image_cache = std.StringHashMap(img.Image).init(allocator),
        };
    }

    pub fn setImage(self: *Context, image: img.Image) void {
        if (self.image_loaded) {
            self.image.deinit(self.allocator);
        }
        self.image = image;
        self.image_loaded = true;
    }

    /// Get or create a temporary buffer of at least the specified size
    pub fn getTempBuffer(self: *Context, min_size: usize) ![]img.color.Rgba32 {
        if (self.temp_buffer) |buf| {
            if (buf.len >= min_size) {
                return buf[0..min_size];
            } else {
                // Free existing buffer if too small
                self.allocator.free(buf);
                self.temp_buffer = null;
            }
        }

        // Allocate new buffer
        const new_buf = try self.allocator.alloc(img.color.Rgba32, min_size);
        self.temp_buffer = new_buf;
        self.temp_buffer_size = min_size;
        return new_buf;
    }

    /// Copy pixels to a temporary buffer, reusing existing allocation if possible
    pub fn copyToTempBuffer(self: *Context, pixels: []const img.color.Rgba32) ![]img.color.Rgba32 {
        const temp_buf = try self.getTempBuffer(pixels.len);
        @memcpy(temp_buf, pixels);
        return temp_buf;
    }

    pub fn deinit(self: *Context) void {
        if (self.image_loaded) {
            self.image.deinit(self.allocator);
            self.image_loaded = false;
        }
        self.allocator.free(self.output_filename);
        if (self.input_filename) |input| {
            self.allocator.free(input);
        }
        if (self.output_directory) |dir| {
            self.allocator.free(dir);
        }
        if (self.temp_buffer) |buf| {
            self.allocator.free(buf);
        }
        if (self.preset_path) |path| {
            self.allocator.free(path);
        }
        // Clean up image cache
        var cache_iter = self.image_cache.iterator();
        while (cache_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.image_cache.deinit();
        // Note: output_extension is not freed as it's usually a string literal
    }

    pub fn setOutput(self: *Context, filename: []const u8) void {
        const ext = std.fs.path.extension(filename);
        var new_filename: []const u8 = undefined;
        if (ext.len > 0) {
            // Keep the full filename including extension
            new_filename = self.allocator.dupe(u8, filename) catch unreachable;
            self.output_extension = ext;
        } else {
            new_filename = self.allocator.dupe(u8, filename) catch unreachable;
        }
        self.allocator.free(self.output_filename);
        self.output_filename = new_filename;
    }

    pub fn setOutputDirectory(self: *Context, directory: []const u8) void {
        if (self.output_directory) |old_dir| {
            self.allocator.free(old_dir);
        }
        self.output_directory = self.allocator.dupe(u8, directory) catch unreachable;
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
