pub const GalleryExample = struct {
    name: []const u8,
    description: []const u8,
    args: []const []const u8,

    pub fn init(name: []const u8, description: []const u8, args: []const []const u8) GalleryExample {
        return .{ .name = name, .description = description, .args = args };
    }
};

pub const individual_modifiers = [_]GalleryExample{
    GalleryExample.init("Invert", "Invert all colors", &[_][]const u8{"invert"}),
    GalleryExample.init("Grayscale", "Convert to grayscale", &[_][]const u8{"grayscale"}),
    GalleryExample.init("Brightness +50", "Increase brightness", &[_][]const u8{ "brightness", "50" }),
    GalleryExample.init("Brightness -30", "Decrease brightness", &[_][]const u8{ "brightness", "-30" }),
    GalleryExample.init("Contrast 1.5", "Increase contrast", &[_][]const u8{ "contrast", "1.5" }),
    GalleryExample.init("Contrast 0.7", "Decrease contrast", &[_][]const u8{ "contrast", "0.7" }),
    GalleryExample.init("Saturation 1.5", "Increase saturation", &[_][]const u8{ "saturation", "1.5" }),
    GalleryExample.init("Saturation 0.5", "Decrease saturation", &[_][]const u8{ "saturation", "0.5" }),
    GalleryExample.init("Gamma 2.2", "Apply gamma correction (brighter)", &[_][]const u8{ "gamma", "2.2" }),
    GalleryExample.init("Gamma 0.45", "Apply gamma correction (darker)", &[_][]const u8{ "gamma", "0.45" }),
    GalleryExample.init("Sepia", "Apply sepia tone effect", &[_][]const u8{"sepia"}),
    GalleryExample.init("Blur", "Apply box blur with kernel size 3", &[_][]const u8{ "blur", "3" }),
    GalleryExample.init("Sharpen", "Sharpen the image", &[_][]const u8{"sharpen"}),
    GalleryExample.init("Flip Horizontal", "Mirror horizontally", &[_][]const u8{ "flip", "horizontal" }),
    GalleryExample.init("Flip Vertical", "Mirror vertically", &[_][]const u8{ "flip", "vertical" }),
    GalleryExample.init("Rotate 45°", "Rotate 45 degrees clockwise", &[_][]const u8{ "rotate", "45" }),
    GalleryExample.init("Resize", "Resize to 200x200", &[_][]const u8{ "resize", "200", "200" }),
    GalleryExample.init("Crop", "Crop to center 200x200", &[_][]const u8{ "crop", "25", "25", "200", "200" }),
};

pub const combinations = [_]GalleryExample{
    GalleryExample.init("Grayscale + Sepia", "Vintage effect", &[_][]const u8{ "grayscale", "sepia" }),
    GalleryExample.init("Brightness + Contrast + Sharpen", "Enhanced image", &[_][]const u8{ "brightness", "30", "contrast", "1.2", "sharpen" }),
    GalleryExample.init("Saturation + Gamma", "Color corrected", &[_][]const u8{ "saturation", "0.3", "gamma", "1.5" }),
    GalleryExample.init("Blur + Sharpen", "Noise reduction with detail enhancement", &[_][]const u8{ "blur", "5", "sharpen" }),
    GalleryExample.init("Sepia + Brightness", "Warm vintage tone", &[_][]const u8{ "sepia", "brightness", "20" }),
};
