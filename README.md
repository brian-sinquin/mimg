
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

### Pre-built Installers (Recommended)

Download the latest release for your platform from the [Releases page](https://github.com/brian-sinquin/mimg/releases):

#### Windows
- **MSI Installer** (`mimg-installer.msi`): Professional installer with automatic PATH configuration and Start Menu shortcuts
- **Portable ZIP** (`mimg-x86_64-windows.zip`): Extract and run without installation

#### macOS
- **DMG Image** (`mimg-*.dmg`): Drag-and-drop installer with visual interface
- **PKG Installer** (`mimg-*.pkg`): Traditional macOS installer package
- **Portable TAR.GZ** (`mimg-x86_64-macos.tar.gz`): Extract and use directly

#### Linux
- **DEB Package** (`mimg_*_amd64.deb`): For Debian, Ubuntu, and derivatives
  ```bash
  sudo dpkg -i mimg_*_amd64.deb
  ```
- **RPM Package** (`mimg-*.rpm`): For Fedora, RHEL, and derivatives
  ```bash
  sudo rpm -i mimg-*.rpm
  ```
- **AppImage** (`mimg-*-x86_64.AppImage`): Universal Linux executable, no installation required
  ```bash
  chmod +x mimg-*-x86_64.AppImage
  ./mimg-*-x86_64.AppImage
  ```
- **Portable TAR.GZ** (`mimg-x86_64-linux.tar.gz`): Extract and use directly

### Build from Source

```bash
git clone https://github.com/brian-sinquin/mimg.git
cd mimg
zig build
```

The compiled binary will be available at `zig-out/bin/mimg`.

### Requirements

- Zig 0.15.1 or later (for building from source)

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
