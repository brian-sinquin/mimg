const std = @import("std");
const types = @import("types.zig");
const utils = @import("utils.zig");
const argument_parser = @import("argument_parser.zig");

// Version is set at build time via build options
pub const VERSION: []const u8 = @import("build_options").version;

// Re-export for backward compatibility
pub const registered_options = argument_parser.all_options;
pub const getOptionName = argument_parser.getOptionName;
pub const printModifiers = argument_parser.printModifiers;
pub const printVersion = argument_parser.printVersion;
pub const processArguments = argument_parser.processArguments;
pub const processArgumentsFromSlice = argument_parser.processArgumentsFromSlice;
pub const matchesOption = argument_parser.matchesOption;
pub const printHelp = argument_parser.printHelp;
pub const printImageInfo = argument_parser.printImageInfo;
pub const getOptions = argument_parser.getOptions;
pub const reportParseError = argument_parser.reportParseError;
pub const applyPreset = argument_parser.applyPreset;
