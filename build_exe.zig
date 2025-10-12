const std = @import("std");

// Import build config for version
const build_config = @import("build.zig.zon");

pub fn createExe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const release_version = std.process.getEnvVarOwned(b.allocator, "RELEASE_VERSION") catch "dev";
    const target_name = b.option([]const u8, "target-name", "Target name for binary") orelse "unknown";

    // Build options for optimization
    // LTO is disabled by default to avoid compilation issues on Linux/macOS CI/CD environments
    // Enable with -Dlto=true for local release builds if desired
    const enable_lto = b.option(bool, "lto", "Enable Link Time Optimization (slower builds, faster runtime)") orelse false;
    const strip_symbols = b.option(bool, "strip", "Strip debug symbols (smaller binary)") orelse (optimize != .Debug);
    const use_static = b.option(bool, "static", "Force static linking") orelse false;

    const exe = b.addExecutable(.{
        .name = b.fmt("{s}-v{s}-{s}", .{ "mimg", release_version, target_name }),
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Enable LTO for better optimization (but slower builds)
    if (enable_lto) {
        exe.want_lto = true;
    }

    // Strip symbols for smaller binaries in release builds
    exe.root_module.strip = strip_symbols;

    // Force static linking if requested
    if (use_static) {
        exe.linkage = .static;
    }

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
