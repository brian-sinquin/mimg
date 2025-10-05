const std = @import("std");

// Helper function to create gallery command for an example
fn createGalleryCommand(b: *std.Build, exe: *std.Build.Step.Compile, example: anytype, gallery_step: *std.Build.Step) void {
    const gallery_cmd = b.addRunArtifact(exe);
    gallery_cmd.addFileArg(b.path("examples/gallery/lena.png"));

    for (example.args) |arg| {
        gallery_cmd.addArg(arg);
    }

    // Generate output filename
    var output_name_buf: [128]u8 = undefined;
    var output_name_len: usize = 0;

    for (example.args, 0..) |arg, i| {
        if (i > 0) {
            @memcpy(output_name_buf[output_name_len .. output_name_len + 1], "_");
            output_name_len += 1;
        }
        @memcpy(output_name_buf[output_name_len .. output_name_len + arg.len], arg);
        output_name_len += arg.len;
    }
    @memcpy(output_name_buf[output_name_len .. output_name_len + 9], "_lena.png");
    output_name_len += 9;

    const output_filename = output_name_buf[0..output_name_len];

    // Build full output path
    var full_path_buf: [256]u8 = undefined;
    const full_path = std.fmt.bufPrint(&full_path_buf, "examples/gallery/output/{s}", .{output_filename}) catch unreachable;

    gallery_cmd.addArg("--output");
    gallery_cmd.addArg(b.dupe(full_path));

    gallery_step.dependOn(&gallery_cmd.step);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const documentation = b.option(bool, "docs", "Generate documentation") orelse false;

    const exe = b.addExecutable(.{
        .name = "mimg",
        .root_module = b.createModule(.{ .root_source_file = b.path("src/main.zig"), .target = target, .optimize = optimize }),
    });

    const zigimg_dependency = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const gallery_step = b.step("gallery", "Generate example gallery");

    // Clean up old gallery files and create output directory
    const cleanup_cmd = b.addSystemCommand(&[_][]const u8{ "cmd", "/c", "if exist examples\\gallery\\output rmdir /s /q examples\\gallery\\output & mkdir examples\\gallery\\output" });
    gallery_step.dependOn(&cleanup_cmd.step);

    // Ensure output directory exists before running commands
    const ensure_output_cmd = b.addSystemCommand(&[_][]const u8{ "cmd", "/c", "if not exist examples\\gallery\\output mkdir examples\\gallery\\output" });
    gallery_step.dependOn(&ensure_output_cmd.step);

    // Gallery generator executable
    const gallery_exe = b.addExecutable(.{
        .name = "gallery_generator",
        .root_module = b.createModule(.{ .root_source_file = b.path("src/gallery/gallery.zig"), .target = target, .optimize = optimize }),
    });

    gallery_exe.step.dependOn(&cleanup_cmd.step);

    gallery_step.dependOn(&gallery_exe.step);
    gallery_step.dependOn(b.getInstallStep());

    // Import gallery data
    const gallery_data = @import("src/gallery/gallery_data.zig");

    // Generate examples using shared data
    inline for (gallery_data.individual_modifiers) |modifier| {
        createGalleryCommand(b, exe, modifier, gallery_step);
    }

    inline for (gallery_data.combinations) |combo| {
        createGalleryCommand(b, exe, combo, gallery_step);
    }

    // Run gallery generator
    const gallery_run = b.addRunArtifact(gallery_exe);
    gallery_step.dependOn(&gallery_run.step);

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
