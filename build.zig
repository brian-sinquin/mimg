const std = @import("std");
const build_exe = @import("build_exe.zig");
const build_gallery = @import("build_gallery.zig");

fn createModuleWithZigimg(b: *std.Build, root_source_file: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
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

    return module;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options
    const documentation = b.option(bool, "docs", "Generate documentation") orelse false;
    const enable_gallery = true; // b.option(bool, "gallery", "Enable gallery generation") orelse false;
    const enable_tests = b.option(bool, "tests", "Enable test building") orelse false;
    const enable_benchmarks = b.option(bool, "benchmarks", "Enable benchmark building") orelse false;
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
        const unit_test_module = createModuleWithZigimg(b, "src/main.zig", target, optimize);

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
        const benchmark_module = createModuleWithZigimg(b, "src/testing/benchmarks.zig", target, optimize);

        const benchmarks = b.addTest(.{
            .root_module = benchmark_module,
        });

        const run_benchmarks = b.addRunArtifact(benchmarks);

        const benchmark_step = b.step("bench", "Run performance benchmarks");
        benchmark_step.dependOn(&run_benchmarks.step);
    }

    // Quick build step for development (always Debug, no LTO)
    const quick_build_step = b.step("quick", "Quick development build (Debug, no LTO)");
    const quick_exe = build_exe.createExe(b, target, .Debug, target_name, false, false, false, null);
    quick_build_step.dependOn(&quick_exe.step);

    // Release build step with all optimizations
    const release_build_step = b.step("release", "Optimized release build");
    const release_exe = build_exe.createExe(b, target, .ReleaseFast, target_name, false, true, false, "avx2");
    release_build_step.dependOn(&release_exe.step);

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
