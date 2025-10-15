const app = @import("app/app.zig");
const testing = @import("testing/tests.zig");
const benchmarks = @import("testing/benchmarks.zig");

// Ensure test modules are included in the build
comptime {
    _ = testing;
    _ = benchmarks;
}

pub fn main() !void {
    try app.run();
}
