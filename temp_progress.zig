/// Simple progress bar for long operations
pub const ProgressBar = struct {
    total: usize,
    current: std.atomic.Value(usize),
    width: usize = 50,

    pub fn init(total: usize) ProgressBar {
        return .{
            .total = total,
            .current = std.atomic.Value(usize).init(0),
        };
    }

    pub fn increment(self: *ProgressBar) void {
        _ = self.current.fetchAdd(1, .monotonic);
    }

    pub fn update(self: *ProgressBar, writer: anytype) !void {
        const current = self.current.load(.monotonic);
        const percentage = @as(f32, @floatFromInt(current)) / @as(f32, @floatFromInt(self.total));
        const filled = @as(usize, @intFromFloat(percentage * @as(f32, @floatFromInt(self.width))));

        try writer.writeAll("\r[");
        for (0..self.width) |i| {
            if (i < filled) {
                try writer.writeAll("=");
            } else if (i == filled) {
                try writer.writeAll(">");
            } else {
                try writer.writeAll(" ");
            }
        }
        try writer.print("] {d}/{d} ({d:.1}%)", .{ current, self.total, percentage * 100 });
        if (current >= self.total) {
            try writer.writeAll("\n");
        }
    }
};
