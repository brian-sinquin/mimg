const std = @import("std");

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

    pub fn update(self: *ProgressBar, current: usize, total: usize) !void {
        const stdout_writer = std.fs.File.stdout().deprecatedWriter();
        const percentage = @as(f32, @floatFromInt(current)) / @as(f32, @floatFromInt(total));
        const filled = @as(usize, @intFromFloat(percentage * @as(f32, @floatFromInt(self.width))));

        try stdout_writer.writeAll("\r[");
        for (0..self.width) |i| {
            if (i < filled) {
                try stdout_writer.writeAll("=");
            } else if (i == filled) {
                try stdout_writer.writeAll(">");
            } else {
                try stdout_writer.writeAll(" ");
            }
        }
        try stdout_writer.print("] {d}/{d} ({d:.1}%)", .{ current, total, percentage * 100 });
        if (current >= total) {
            try stdout_writer.writeAll("\n");
        }
    }
};
