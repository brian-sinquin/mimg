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
    GalleryExample.init("Blue Tint", "Cool blue colorization", &[_][]const u8{ "colorize", "#6496FF", "0.4" }),
    GalleryExample.init("Oil Painting", "Artistic oil paint effect", &[_][]const u8{ "oil-painting", "3" }),
    GalleryExample.init("Tilt Shift", "Advanced miniature depth of field effect", &[_][]const u8{ "tilt-shift", "5.0", "0.5", "0.08" }),
    GalleryExample.init("Round Corners", "Smooth rounded corners", &[_][]const u8{ "round-corners", "25" }),
    GalleryExample.init("Rotate 30°", "Rotate 30 degrees", &[_][]const u8{ "rotate", "30" }),
    GalleryExample.init("Flip Horizontal", "Mirror horizontally", &[_][]const u8{ "flip", "horizontal" }),
};

pub const combinations = [_]GalleryExample{
    GalleryExample.init("Vintage Portrait", "Sepia tone with a gentle vignette for a classic look", &[_][]const u8{ "sepia", "contrast", "1.1", "vignette", "0.45" }),
    GalleryExample.init("Pop Art", "Bold posterization with punchy color and contrast", &[_][]const u8{ "gaussian-blur", "2.0", "posterize", "5", "edge-enhancement", "0.3", "saturation", "1.6", "contrast", "0.5" }),
    GalleryExample.init("Cyberpunk", "Cool neon tint, soft glow, and crisp edges", &[_][]const u8{ "colorize", "#00A8FF", "0.55", "glow", "0.8", "12", "edge-enhancement", "1.35", "contrast", "1.15", "vignette", "0.15" }),
    GalleryExample.init("Dreamy Portrait", "Soft painterly look with a gentle bloom", &[_][]const u8{ "oil-painting", "3", "glow", "0.35", "12", "exposure", "0.15", "saturation", "1.1", "vignette", "0.25" }),
    GalleryExample.init("Film Noir", "Moody black and white with contrast and vignette", &[_][]const u8{ "grayscale", "contrast", "1.9", "sharpen", "vignette", "0.65", "noise", "0.06" }),
    GalleryExample.init("Artistic Sketch", "Edge outlines with minimal palette", &[_][]const u8{ "edge-detect", "invert", "posterize", "3", "contrast", "1.2" }),
    GalleryExample.init("Neon Glow", "Bright edges with colorful radiance", &[_][]const u8{ "edge-enhancement", "1.3", "glow", "0.55", "12", "saturation", "1.2", "vignette", "0.10" }),
    GalleryExample.init("Solar Dream", "Radiant duotone with soft glow and high vibrance", &[_][]const u8{ "duotone", "#FFD700", "#FF69B4", "glow", "0.5", "10", "vibrance", "0.8", "contrast", "1.2" }),
    GalleryExample.init("Retro Fade", "Muted colors, film grain, and faded vignette for a vintage look", &[_][]const u8{ "posterize", "7", "saturation", "0.7", "noise", "0.12", "vignette", "0.35", "contrast", "0.8" }),
    GalleryExample.init("Arctic Mist", "Cool blue tint, gaussian blur, and crisp edges for frosty effect", &[_][]const u8{ "colorize", "#AEEFFF", "0.4", "gaussian-blur", "2.5", "edge-enhancement", "1.1", "contrast", "1.05" }),
    GalleryExample.init("Nightlife", "Vivid colors, tilt-shift, and high contrast for a dramatic city vibe", &[_][]const u8{ "saturation", "1.7", "tilt-shift", "6.5", "0.5", "0.07", "contrast", "1.3", "vignette", "0.22" }),
};
