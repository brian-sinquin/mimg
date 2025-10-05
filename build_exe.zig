const std = @import("std");

// Import build config for version
const build_config = @import("build.zig.zon");

pub fn createExe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const release_version = std.process.getEnvVarOwned(b.allocator, "RELEASE_VERSION") catch "dev";
    const target_name = b.option([]const u8, "target-name", "Target name for binary") orelse "unknown";

    const exe = b.addExecutable(.{
        .name = b.fmt("{s}-v{s}-{s}", .{ "mimg", release_version, target_name }),
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Add version constant to the module
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", build_config.version);
    exe.root_module.addOptions("build_options", build_options);

    const zigimg_dependency = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

    b.installArtifact(exe);

    return exe;
}
