const std = @import("std");

// Import build config for version
const build_config = @import("../build.zig.zon");

pub fn createExe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, target_name: []const u8, enable_lto: bool, strip_symbols: bool, use_static: bool, cpu_features: ?[]const u8) *std.Build.Step.Compile {
    const release_version = std.process.getEnvVarOwned(b.allocator, "RELEASE_VERSION") catch "dev";

    // Modify target for CPU features if specified
    var modified_target = target;
    if (cpu_features) |features| {
        var cpu = target.result.cpu;
        if (std.mem.eql(u8, features, "avx2")) {
            cpu.features.addFeature(@intFromEnum(std.Target.x86.Feature.avx2));
        } else if (std.mem.eql(u8, features, "sse4_2")) {
            cpu.features.addFeature(@intFromEnum(std.Target.x86.Feature.sse4_2));
        }
        // Add more CPU features as needed
        modified_target = b.resolveTargetQuery(.{
            .cpu_arch = target.result.cpu.arch,
            .os_tag = target.result.os.tag,
            .abi = target.result.abi,
            .cpu_model = .{ .explicit = cpu.model },
            .cpu_features_add = cpu.features,
        });
    }

    const exe = b.addExecutable(.{
        .name = b.fmt("{s}-v{s}-{s}", .{ "mimg", release_version, target_name }),
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = modified_target,
            .optimize = optimize,
        }),
    });

    // Enable sanitizers in debug mode (TODO: implement when API is available)
    // if (enable_sanitizers and optimize == .Debug) {
    //     exe.sanitize_address = true;
    //     exe.sanitize_thread = true;
    //     exe.sanitize_undefined_behavior = true;
    // }

    // Enable LTO for better optimization (but slower builds)
    exe.want_lto = enable_lto;

    // Strip symbols for smaller binaries in release builds
    exe.root_module.strip = strip_symbols;

    // Force static linking if requested
    if (use_static) {
        exe.linkage = .static;
    }

    // Add version constant to the module
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", build_config.version);
    build_options.addOption(bool, "include_benchmarks", false);
    exe.root_module.addOptions("build_options", build_options);

    const zigimg_dependency = b.dependency("zigimg", .{
        .target = modified_target,
        .optimize = optimize,
    });

    exe.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

    b.installArtifact(exe);

    return exe;
}
