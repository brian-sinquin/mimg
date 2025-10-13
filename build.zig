const std = @import("std");
const build_exe = @import("build/build_exe.zig");
const build_gallery = @import("build/build_gallery.zig");

fn createModuleWithZigimg(b: *std.Build, root_source_file: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, include_benchmarks: bool) *std.Build.Module {
    const module = b.createModule(.{
        .root_source_file = b.path(root_source_file),
        .target = target,
        .optimize = optimize,
    });

    const zigimg_dependency = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });
    module.addImport("zigimg", zigimg_dependency.module("zigimg"));

    // Add build options
    const options = b.addOptions();
    options.addOption(bool, "include_benchmarks", include_benchmarks);
    module.addImport("build_options", options.createModule());

    return module;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options
    // Gallery is always generated as a build step; no flag required
    // Tests and benchmarks are always enabled; no flags required
    const target_name = b.option([]const u8, "target-name", "Target name for binary") orelse "unknown";
    const enable_lto = b.option(bool, "lto", "Enable Link Time Optimization (slower builds, faster runtime)") orelse false;
    const strip_symbols = b.option(bool, "strip", "Strip debug symbols (smaller binary)") orelse (optimize != .Debug);
    const use_static = b.option(bool, "static", "Force static linking") orelse false;
    const cpu_features = b.option([]const u8, "cpu-features", "CPU features to enable (avx2, sse4_2)") orelse null;

    // Create main executable
    const exe = build_exe.createExe(b, target, optimize, target_name, enable_lto, strip_symbols, use_static, cpu_features);

    // Run step
    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    // Don't depend on global install to avoid building unrelated artifacts

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Gallery generation (always enabled, not an executable)
    _ = build_gallery.setupGallery(b, target, optimize, exe);

    // Always define test step
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    // Add separate test for our test file with proper dependencies
    const unit_test_module = createModuleWithZigimg(b, "src/main.zig", target, optimize, false);
    const unit_tests = b.addTest(.{
        .root_module = unit_test_module,
    });
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&exe_tests.step);
    test_step.dependOn(&unit_tests.step);

    // ...existing code...

    // ...existing code...

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    clean_step.makeFn = struct {
        pub fn make(_: *std.Build.Step, options: std.Build.Step.MakeOptions) !void {
            _ = options;
            const fs = std.fs;
            const cwd = fs.cwd();
            _ = cwd.deleteTree("zig-out") catch {};
            _ = cwd.deleteTree("zig-cache") catch {};
        }
    }.make;
}
