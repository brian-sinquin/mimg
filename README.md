# mimg

[![mimg](https://github.com/brian-sinquin/mimg/actions/workflows/dev.yml/badge.svg)](https://github.com/brian-sinquin/mimg/actions/workflows/dev.yml) [![GitHub release (latest by date)](https://img.shields.io/github/v/release/brian-sinquin/mimg?label=ver)](https://github.com/brian-sinquin/mimg/releases/latest)

**Fast, SIMD-optimized command-line image processing tool written in Zig**

## ✨ Features

- 🚀 **High Performance**: SIMD-optimized operations with 1000+ MPixels/sec throughput
- 🎨 **Rich Effects**: 30+ modifiers including color adjustments, filters, and transforms
- 🔗 **Chain Operations**: Combine multiple effects in a single command
- 📝 **Presets**: Save and reuse complex processing chains
- 🖼️ **Multiple Formats**: PNG, TGA, QOI, PAM, PBM, PGM, PPM, PCX support
- 🧵 **Multithreaded**: Efficient batch processing of multiple images
- 📚 **Library API**: Use as a Zig library for programmatic processing

## 🚀 Quick Start

```bash
# Install from releases or build from source
zig build

# Apply a single effect
zig build run -- input.png brightness 20 -o output.png

# Chain multiple effects
zig build run -- input.png brightness 20 saturation 1.3 sharpen -o output.png

# Use a preset for complex processing
zig build run -- input.png --preset vintage.preset -o output.png
```

## 📖 Documentation

- **[Installation & Building](docs/installation.md)** - Get started with mimg
- **[Available Modifiers](docs/modifiers.md)** - Complete list of all effects and transforms
- **[Presets](docs/presets.md)** - Save and reuse processing chains
- **[Usage Examples](docs/examples.md)** - Practical examples and use cases
- **[Performance](docs/performance.md)** - Benchmarks and optimization tips
- **[File Formats](docs/formats.md)** - Supported input/output formats
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Library API](docs/library.md)** - Use mimg programmatically in Zig

## 📊 Performance Highlights

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Grayscale | 1078 MPixels/sec | Simple luminance calculation |
| Invert Colors | 1025 MPixels/sec | Fast bitwise operations |
| Brightness | 831 MPixels/sec | SIMD-optimized |
| Gaussian Blur | 72 MPixels/sec | Separable kernel |
| Rotate 90° | 1334 MPixels/sec | Pixel rearrangement |

*Benchmarks on Intel i7-9750H with 256×256 images*

## 🛠️ Requirements

- Zig 0.15.1 or later
- No external dependencies

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
