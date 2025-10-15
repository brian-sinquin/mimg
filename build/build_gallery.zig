const std = @import("std");
const build_utils = @import("build_utils.zig");

// Helper function to create gallery command for an example
pub fn createGalleryCommand(b: *std.Build, exe: *std.Build.Step.Compile, example: anytype, gallery_step: *std.Build.Step) void {
    const gallery_cmd = b.addRunArtifact(exe);
    gallery_cmd.addFileArg(b.path("examples/gallery/lena.png"));

    for (example.args) |arg| {
        gallery_cmd.addArg(arg);
    }

    // Generate output filename
    const output_filename = build_utils.generateGalleryFilename(b.allocator, example.args) catch @panic("Failed to generate gallery filename");

    // Use --output-dir and --output separately
    gallery_cmd.addArg("--output-dir");
    gallery_cmd.addArg("examples/gallery/output");
    gallery_cmd.addArg("--output");
    gallery_cmd.addArg(output_filename);

    gallery_step.dependOn(&gallery_cmd.step);
}

pub fn setupGallery(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, exe: *std.Build.Step.Compile) *std.Build.Step {
    const gallery_step = b.step("gallery", "Generate example gallery");

    // Clean up old gallery files and create output directory
    const cleanup_step = b.step("cleanup_gallery_output", "Clean up gallery output directory");
    cleanup_step.makeFn = struct {
        pub fn make(_: *std.Build.Step, options: std.Build.Step.MakeOptions) !void {
            _ = options;
            const fs = std.fs;
            const cwd = fs.cwd();
            const output_path = "examples/gallery/output";
            // Try to delete the directory if it exists
            _ = cwd.deleteTree(output_path) catch {};
            // Recreate the directory
            try cwd.makePath(output_path);
        }
    }.make;
    gallery_step.dependOn(cleanup_step);

    // Gallery generator executable
    const gallery_exe = b.addExecutable(.{
        .name = "gallery_generator",
        .root_module = b.createModule(.{ .root_source_file = b.path("src/gallery/gallery.zig"), .target = target, .optimize = optimize }),
    });

    gallery_exe.step.dependOn(cleanup_step);

    gallery_step.dependOn(&gallery_exe.step);
    // gallery_step.dependOn(b.getInstallStep()); // Not needed since we run from build artifacts

    // Import gallery data
    const gallery_data = @import("../src/gallery/gallery_data.zig");

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

    return gallery_step;
}
