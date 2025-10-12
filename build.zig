const std = @import("std");
const build_exe = @import("build_exe.zig");
const build_gallery = @import("build_gallery.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options
    const documentation = b.option(bool, "docs", "Generate documentation") orelse false;
    const enable_gallery = true; // b.option(bool, "gallery", "Enable gallery generation") orelse false;
    const enable_tests = b.option(bool, "tests", "Enable test building") orelse false;
    const enable_benchmarks = b.option(bool, "benchmarks", "Enable benchmark building") orelse false;

    // Create main executable
    const exe = build_exe.createExe(b, target, optimize);

    // Run step
    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Gallery generation (only if enabled)
    if (enable_gallery) {
        _ = build_gallery.setupGallery(b, target, optimize, exe);
    }

    // Documentation
    if (documentation) {
        std.log.warn("Documentation generation is not yet supported on this Zig release.", .{});
    }

    // Tests (only if enabled)
    if (enable_tests) {
        const exe_tests = b.addTest(.{
            .root_module = exe.root_module,
        });

        // Add separate test for our test file with proper dependencies
        const unit_test_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
            .target = target,
            .optimize = optimize,
        });

        // Add zigimg dependency to the test module
        const zigimg_dependency = b.dependency("zigimg", .{
            .target = target,
            .optimize = optimize,
        });
        unit_test_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

        const unit_tests = b.addTest(.{
            .root_module = unit_test_module,
        });

        const run_exe_tests = b.addRunArtifact(exe_tests);
        const run_unit_tests = b.addRunArtifact(unit_tests);

        const test_step = b.step("test", "Run tests");
        test_step.dependOn(&run_exe_tests.step);
        test_step.dependOn(&run_unit_tests.step);
    }

    // Benchmarks (only if enabled)
    if (enable_benchmarks) {
        const benchmark_module = b.createModule(.{
            .root_source_file = b.path("src/benchmarks.zig"),
            .target = target,
            .optimize = optimize,
        });

        // Add zigimg dependency to the benchmark module
        const zigimg_dependency = b.dependency("zigimg", .{
            .target = target,
            .optimize = optimize,
        });
        benchmark_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

        const benchmarks = b.addTest(.{
            .root_module = benchmark_module,
        });

        const run_benchmarks = b.addRunArtifact(benchmarks);

        const benchmark_step = b.step("bench", "Run performance benchmarks");
        benchmark_step.dependOn(&run_benchmarks.step);
    }

    // Quick build step for development
    const quick_build_step = b.step("quick", "Quick development build (Debug, no LTO)");
    quick_build_step.dependOn(b.getInstallStep());
}
