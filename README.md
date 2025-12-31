
# mimg

A high-performance command-line image processing tool written in Zig.

**[View Documentation & Gallery](https://mimg.brian-sinquin.fr)**

## Features

- Fast image processing with support for multiple formats (PNG, JPEG, BMP, QOI)
- Chainable modifiers for complex transformations
- Built-in presets for common operations
- Comprehensive set of filters and effects
- Zero external dependencies

## Installation

### Build from Source

```bash
git clone https://github.com/brian-sinquin/mimg.git
cd mimg
zig build
```

The compiled binary will be available at `zig-out/bin/mimg`.

### Requirements

- Zig 0.15.1 or later

## Usage

### Basic Syntax

```bash
mimg <input> [modifiers...] [-o output]
```

### Examples

```bash
# Adjust brightness
mimg input.png brightness 20 -o output.png

# Chain multiple modifiers
mimg input.png grayscale blur 5 -o output.png

# Use a preset
mimg input.png preset vintage -o output.png
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Available Modifiers](docs/modifiers.md)
- [Usage Examples](docs/examples.md)
- [Presets](docs/presets.md)
- [Supported Formats](docs/formats.md)

## Development

### Build the Website

Generate the documentation website with gallery examples:

```bash
zig build website
```

Output will be in `website/zig-out/`.

### Run Tests

```bash
zig build test
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Documentation](https://mimg.brian-sinquin.fr)
- [Gallery](https://mimg.brian-sinquin.fr/gallery)
- [GitHub Repository](https://github.com/brian-sinquin/mimg)
