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
    GalleryExample.init("Gaussian Blur", "Apply Gaussian blur with sigma 2.0", &[_][]const u8{ "gaussian-blur", "2.0" }),
    GalleryExample.init("Emboss", "Apply emboss effect", &[_][]const u8{"emboss"}),
    GalleryExample.init("Vignette", "Apply vignette effect", &[_][]const u8{ "vignette", "0.5" }),
    GalleryExample.init("Posterize", "Reduce to 8 color levels", &[_][]const u8{ "posterize", "8" }),
    GalleryExample.init("Hue Shift", "Shift hue by 15 degrees", &[_][]const u8{ "hue-shift", "15" }),
    GalleryExample.init("Median Filter", "Apply 3x3 median filter", &[_][]const u8{ "median-filter", "3" }),
    GalleryExample.init("Threshold", "Convert to pure black and white", &[_][]const u8{ "threshold", "128" }),
    GalleryExample.init("Solarize", "Solarize effect with threshold", &[_][]const u8{ "solarize", "128" }),
    GalleryExample.init("Edge Detect", "Detect edges using Sobel operator", &[_][]const u8{"edge-detect"}),
    GalleryExample.init("Pixelate", "Apply pixelation effect", &[_][]const u8{ "pixelate", "10" }),
    GalleryExample.init("Noise", "Add random noise to image", &[_][]const u8{ "noise", "0.1" }),
    GalleryExample.init("Exposure +1", "Increase exposure by 1 EV", &[_][]const u8{ "exposure", "1.0" }),
    GalleryExample.init("Exposure -1", "Decrease exposure by 1 EV", &[_][]const u8{ "exposure", "-1.0" }),
    GalleryExample.init("Vibrance", "Boost vibrance (smart saturation)", &[_][]const u8{ "vibrance", "0.5" }),
    GalleryExample.init("Equalize", "Histogram equalization for contrast", &[_][]const u8{"equalize"}),
    GalleryExample.init("Colorize Blue", "Tint with blue color", &[_][]const u8{ "colorize", "50", "100", "200", "0.6" }),
    GalleryExample.init("Duotone", "Purple to yellow duotone effect", &[_][]const u8{ "duotone", "75", "30", "120", "255", "220", "100" }),
    GalleryExample.init("Oil Painting", "Artistic oil painting effect", &[_][]const u8{ "oil-painting", "3" }),
    GalleryExample.init("Flip Horizontal", "Mirror horizontally", &[_][]const u8{ "flip", "horizontal" }),
    GalleryExample.init("Flip Vertical", "Mirror vertically", &[_][]const u8{ "flip", "vertical" }),
    GalleryExample.init("Rotate 45°", "Rotate 45 degrees clockwise", &[_][]const u8{ "rotate", "45" }),
    GalleryExample.init("Resize", "Resize to 200x200", &[_][]const u8{ "resize", "200", "200" }),
    GalleryExample.init("Crop", "Crop to center 200x200", &[_][]const u8{ "crop", "25", "25", "200", "200" }),
};

pub const combinations = [_]GalleryExample{
    // ML Dataset Preparation - Data Augmentation
    GalleryExample.init("Data Augmentation", "Flip + brightness variation for ML training data", &[_][]const u8{ "flip", "horizontal", "brightness", "20" }),
    GalleryExample.init("Geometric Augmentation", "Rotation + crop for dataset variety", &[_][]const u8{ "rotate", "15", "crop", "50", "50", "200", "200" }),

    // Photography - Color Grading
    GalleryExample.init("Vintage Portrait", "Sepia tone with vignette for classic look", &[_][]const u8{ "sepia", "vignette", "0.4" }),
    GalleryExample.init("Color Grading", "Brightness + saturation for natural enhancement", &[_][]const u8{ "brightness", "15", "saturation", "1.2" }),

    // Image Processing - Enhancement Pipeline
    GalleryExample.init("Detail Enhancement", "Contrast boost + sharpening for crisp images", &[_][]const u8{ "contrast", "1.2", "sharpen" }),
    GalleryExample.init("Noise Reduction", "Gaussian blur + median filter pipeline", &[_][]const u8{ "gaussian-blur", "1.0", "median-filter", "3" }),

    // Art & Graphics - Creative Effects
    GalleryExample.init("Graphic Art", "Posterize + emboss for artistic rendering", &[_][]const u8{ "posterize", "12", "emboss" }),
    GalleryExample.init("Monitor Calibration", "Gamma correction + contrast adjustment", &[_][]const u8{ "gamma", "2.2", "contrast", "1.1" }),

    // New Creative Effects
    GalleryExample.init("Stylized Edges", "Edge detection + posterize for graphic novel style", &[_][]const u8{ "edge-detect", "posterize", "6" }),
    GalleryExample.init("Retro Game", "Pixelate + posterize for retro gaming aesthetic", &[_][]const u8{ "pixelate", "8", "posterize", "16" }),
    GalleryExample.init("Film Grain", "Solarize + noise for vintage film look", &[_][]const u8{ "solarize", "180", "noise", "0.05" }),

    // Photography Effects
    GalleryExample.init("HDR Look", "Equalize + vibrance for HDR-style enhancement", &[_][]const u8{ "equalize", "vibrance", "0.3" }),
    GalleryExample.init("Dreamy Portrait", "Oil painting + vibrance for soft romantic look", &[_][]const u8{ "oil-painting", "2", "vibrance", "0.4" }),
    GalleryExample.init("Cinematic Grade", "Duotone + exposure for film-like color", &[_][]const u8{ "duotone", "20", "40", "80", "255", "200", "150", "exposure", "0.2" }),
};
