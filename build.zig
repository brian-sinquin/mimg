const std = @import("std");
const build_exe = @import("build_exe.zig");
const build_gallery = @import("build_gallery.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const documentation = b.option(bool, "docs", "Generate documentation") orelse false;

    // Create main executable
    const exe = build_exe.createExe(b, target, optimize);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Setup gallery generation
    // _ = build_gallery.setupGallery(b, target, optimize, exe);

    if (documentation) {
        std.log.warn("Documentation generation is not yet supported on this Zig release.", .{});
    }

    // const exe_tests = b.addTest(.{
    //     .root_module = exe.root_module,
    // });

    // const run_exe_tests = b.addRunArtifact(exe_tests);

    // const test_step = b.step("test", "Run tests");
    // test_step.dependOn(&run_exe_tests.step);
}
