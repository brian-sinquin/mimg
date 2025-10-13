const app = @import("app/app.zig");
const testing = @import("testing/tests.zig");

pub fn main() !void {
    try app.run();
}
