pub const GalleryExample = struct {
    name: []const u8,
    description: []const u8,
    args: []const []const u8,

    pub fn init(name: []const u8, description: []const u8, args: []const []const u8) GalleryExample {
        return .{ .name = name, .description = description, .args = args };
    }
};

pub const individual_modifiers = [_]GalleryExample{
    GalleryExample.init("Invert", "Invert all colors for negative effect", &[_][]const u8{"invert"}),
    GalleryExample.init("Grayscale", "Convert to grayscale", &[_][]const u8{"grayscale"}),
    GalleryExample.init("Sepia", "Classic sepia tone effect", &[_][]const u8{"sepia"}),
    GalleryExample.init("Bright +40", "Increase brightness", &[_][]const u8{ "brightness", "40" }),
    GalleryExample.init("High Contrast", "Boost contrast for dramatic effect", &[_][]const u8{ "contrast", "1.8" }),
    GalleryExample.init("Vibrant Colors", "Boost saturation for vibrant colors", &[_][]const u8{ "saturation", "1.6" }),
    GalleryExample.init("Gamma Correct", "Apply gamma correction", &[_][]const u8{ "gamma", "1.8" }),
    GalleryExample.init("Gaussian Blur", "Soft gaussian blur effect", &[_][]const u8{ "gaussian-blur", "3.0" }),
    GalleryExample.init("Sharp Focus", "Sharpen image details", &[_][]const u8{"sharpen"}),
    GalleryExample.init("Emboss 3D", "3D emboss effect", &[_][]const u8{ "emboss", "1.5" }),
    GalleryExample.init("Color Emboss", "Emboss with color preservation", &[_][]const u8{ "color-emboss", "1.2" }),
    GalleryExample.init("Soft Glow", "Add ethereal glow effect", &[_][]const u8{ "glow", "0.7", "12" }),
    GalleryExample.init("Vignette", "Dark corners vignette", &[_][]const u8{ "vignette", "0.6" }),
    GalleryExample.init("Posterize", "Reduce to 6 color levels", &[_][]const u8{ "posterize", "6" }),
    GalleryExample.init("Hue Shift", "Shift colors by 90 degrees", &[_][]const u8{ "hue-shift", "90" }),
    GalleryExample.init("Edge Detection", "Detect image edges", &[_][]const u8{"edge-detect"}),
    GalleryExample.init("Pixelate", "8-bit pixel effect", &[_][]const u8{ "pixelate", "8" }),
    GalleryExample.init("Film Noise", "Add vintage film grain", &[_][]const u8{ "noise", "0.15" }),
    GalleryExample.init("High Exposure", "Overexposed bright effect", &[_][]const u8{ "exposure", "1.2" }),
    GalleryExample.init("Vibrance", "Smart saturation boost", &[_][]const u8{ "vibrance", "0.6" }),
    GalleryExample.init("Equalize", "Histogram equalization", &[_][]const u8{"equalize"}),
    GalleryExample.init("Blue Tint", "Cool blue colorization", &[_][]const u8{ "colorize", "100", "150", "255", "0.4" }),
    GalleryExample.init("Oil Painting", "Artistic oil paint effect", &[_][]const u8{ "oil-painting", "3" }),
    GalleryExample.init("Tilt Shift", "Miniature depth of field", &[_][]const u8{ "tilt-shift", "2.5", "0.5", "0.25" }),
    GalleryExample.init("Round Corners", "Smooth rounded corners", &[_][]const u8{ "round-corners", "25" }),
    GalleryExample.init("Rotate 30°", "Rotate 30 degrees", &[_][]const u8{ "rotate", "30" }),
    GalleryExample.init("Flip Horizontal", "Mirror horizontally", &[_][]const u8{ "flip", "horizontal" }),
};

pub const combinations = [_]GalleryExample{
    GalleryExample.init("Vintage Portrait", "Sepia tone with soft vignette for classic look", &[_][]const u8{ "sepia", "vignette", "0.5" }),
    GalleryExample.init("Pop Art", "High contrast posterize with vibrant colors", &[_][]const u8{ "posterize", "6", "saturation", "2.0", "contrast", "1.5" }),
    GalleryExample.init("Cyberpunk", "Neon blue glow with edge enhancement", &[_][]const u8{ "colorize", "0", "100", "255", "0.6", "glow", "0.8", "15", "edge-enhancement", "1.2" }),
    GalleryExample.init("Comic Book", "Bold edges with reduced colors and high saturation", &[_][]const u8{ "edge-detect", "posterize", "5", "saturation", "1.8" }),
    GalleryExample.init("Dreamy Portrait", "Soft oil painting with gentle glow", &[_][]const u8{ "oil-painting", "3", "glow", "0.4", "10" }),
    GalleryExample.init("Golden Hour", "Warm gradient with exposure boost", &[_][]const u8{ "gradient-linear", "#ff8c00", "#ffd700", "45", "0.6", "exposure", "0.8" }),
    GalleryExample.init("Retro Game", "Pixelated with reduced color palette", &[_][]const u8{ "pixelate", "12", "posterize", "8" }),
    GalleryExample.init("Film Noir", "High contrast black and white with vignette", &[_][]const u8{ "grayscale", "contrast", "2.0", "vignette", "0.7" }),
    GalleryExample.init("Miniature Model", "Tilt-shift with vibrant colors for toy effect", &[_][]const u8{ "tilt-shift", "2.5", "0.5", "0.3", "vibrance", "0.8" }),
    GalleryExample.init("Artistic Sketch", "Edge detection with inverted posterize", &[_][]const u8{ "edge-detect", "invert", "posterize", "4" }),
    GalleryExample.init("Neon Glow", "Bright edge enhancement with colorful glow", &[_][]const u8{ "edge-enhancement", "1.5", "glow", "0.9", "20", "saturation", "1.4" }),
    GalleryExample.init("Abstract Art", "Color emboss with dramatic HSL shift", &[_][]const u8{ "color-emboss", "1.3", "adjust-hsl", "120", "1.8", "1.2" }),
};
