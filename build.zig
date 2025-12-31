const std = @import("std");
const gallery_data = @import("src/gallery/gallery_data.zig");

/// Generate sanitized filename from command args for gallery examples
fn generateGalleryFilename(allocator: std.mem.Allocator, args: []const []const u8) ![]const u8 {
    var filename = try std.ArrayList(u8).initCapacity(allocator, 256);
    defer filename.deinit(allocator);

    for (args, 0..) |arg, i| {
        if (i > 0) try filename.append(allocator, '_');
        for (arg) |char| {
            switch (char) {
                '.' => try filename.append(allocator, '_'),
                '#' => try filename.appendSlice(allocator, "0x"),
                else => try filename.append(allocator, char),
            }
        }
    }
    try filename.appendSlice(allocator, "_lena.png");
    return try filename.toOwnedSlice(allocator);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get release version from environment or use "dev"
    const release_version = std.process.getEnvVarOwned(b.allocator, "RELEASE_VERSION") catch "dev";

    // Always use simple name "mimg" for the executable
    const exe_name = "mimg";

    // Build main executable
    const exe = b.addExecutable(.{
        .name = exe_name,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const zigimg_dep = b.dependency("zigimg", .{ .target = target, .optimize = optimize });
    exe.root_module.addImport("zigimg", zigimg_dep.module("zigimg"));

    const options = b.addOptions();
    options.addOption([]const u8, "version", "0.1.5");
    options.addOption(bool, "include_benchmarks", false);
    exe.root_module.addImport("build_options", options.createModule());

    b.installArtifact(exe);

    // Run step
    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the app").dependOn(&run_cmd.step);

    // Website generation
    const website_step = b.step("website", "Generate website with gallery");
    setupWebsiteGeneration(b, exe, website_step);

    // Serve website
    const serve_step = b.step("serve", "Serve website locally on port 8080");
    const serve_cmd = b.addSystemCommand(&.{ "python", "-m", "http.server", "8080", "--directory", "website/zig-out" });
    serve_step.dependOn(website_step);
    serve_step.dependOn(&serve_cmd.step);

    // Tests
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    unit_tests.root_module.addImport("zigimg", zigimg_dep.module("zigimg"));
    unit_tests.root_module.addImport("build_options", options.createModule());

    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}

fn setupWebsiteGeneration(b: *std.Build, exe: *std.Build.Step.Compile, website_step: *std.Build.Step) void {
    // Clean and create gallery directory
    const cleanup = b.step("cleanup_website_gallery", "Clean website gallery");
    cleanup.makeFn = struct {
        fn make(_: *std.Build.Step, _: std.Build.Step.MakeOptions) !void {
            const cwd = std.fs.cwd();
            _ = cwd.deleteTree("website/static/gallery") catch {};
            try cwd.makePath("website/static/gallery");
        }
    }.make;
    website_step.dependOn(cleanup);

    // Generate gallery images
    const gen_step = b.step("gen_website_images", "Generate website gallery images");
    inline for (gallery_data.individual_modifiers ++ gallery_data.combinations) |example| {
        const cmd = b.addRunArtifact(exe);
        cmd.addFileArg(b.path("lena.png"));
        for (example.args) |arg| cmd.addArg(arg);
        const filename = generateGalleryFilename(b.allocator, example.args) catch @panic("filename generation failed");
        cmd.addArgs(&.{ "--output-dir", "website/static/gallery", "--output", filename });
        gen_step.dependOn(&cmd.step);
    }
    gen_step.dependOn(cleanup);

    // Build Zine website directly
    const zine_dep = b.dependency("zine", .{ .optimize = .ReleaseFast });
    const zine_exe = zine_dep.artifact("zine");

    const zine_cmd = b.addRunArtifact(zine_exe);
    zine_cmd.addArgs(&.{ "release", "--force", "--output" });
    zine_cmd.addDirectoryArg(b.path("website/zig-out"));
    zine_cmd.setCwd(b.path("website"));
    zine_cmd.step.dependOn(gen_step);

    website_step.dependOn(&zine_cmd.step);
}
