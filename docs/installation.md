# Installation & Building

## Requirements

- **Zig 0.15.1 or later**
- No external dependencies required

## Quick Install

### From Pre-built Binaries

Download the latest release for your platform from [GitHub Releases](https://github.com/brian-sinquin/mimg/releases):

- **Linux**: `mimg-v*-x86_64-linux`
- **macOS**: `mimg-v*-x86_64-macos`
- **Windows**: `mimg-v*-x86_64-windows.exe`

Make the binary executable and add to your PATH:

```bash
# Linux/macOS
chmod +x mimg-v*-x86_64-linux
sudo mv mimg-v*-x86_64-linux /usr/local/bin/mimg

# Windows - just move to a directory in your PATH
```

### From Source

```bash
# Clone the repository
git clone https://github.com/brian-sinquin/mimg.git
cd mimg

# Build the project
zig build

# The binary will be in zig-out/bin/
./zig-out/bin/mimg --help
```

## Building Options

### Debug Build (Default)
```bash
zig build
```

### Release Builds
```bash
# Optimized release build
zig build -Doptimize=ReleaseFast

# Safe release build (recommended for production)
zig build -Doptimize=ReleaseSafe
```

### Cross-Compilation
```bash
# Build for different targets
zig build -Dtarget=x86_64-linux
zig build -Dtarget=x86_64-macos
zig build -Dtarget=x86_64-windows
```

### Build Artifacts

After building, you'll find:
- **Binary**: `zig-out/bin/mimg` (or `mimg.exe` on Windows)
- **Test Results**: Run `zig build test` to verify functionality

## Development Setup

### Running Tests
```bash
# Run all tests
zig build test

# Run tests with verbose output
zig build test -Dverbose
```

### Generating Gallery
```bash
# Generate visual examples
zig build gallery
```

This creates example images in `examples/gallery/output/` and documentation in `examples/gallery/gallery.md`.

### Development Workflow
```bash
# Make changes to source
# Test your changes
zig build test

# Build and test manually
zig build run -- input.png brightness 20 -o output.png

# Generate gallery to verify visual results
zig build gallery
```

## Troubleshooting Build Issues

### Zig Version Problems
```bash
# Check your Zig version
zig version

# Should be 0.15.1 or later
# If too old, update Zig from https://ziglang.org/download/
```

### Dependency Issues
```bash
# Clean and rebuild dependencies
rm -rf .zig-cache/
zig build
```

### Platform-Specific Issues

**Linux:**
- Ensure you have basic build tools installed
- Some distributions may need additional packages

**macOS:**
- Xcode Command Line Tools required: `xcode-select --install`

**Windows:**
- Use the official Zig Windows build
- Or use WSL for Linux-like environment

## Updating

To update to the latest version:

```bash
# If installed from source
cd /path/to/mimg
git pull
zig build

# If using pre-built binaries
# Download latest release from GitHub
```

## Next Steps

Once installed, check out:
- [Available Modifiers](modifiers.md) - Learn about all image processing effects
- [Usage Examples](examples.md) - See practical examples
- [Presets](presets.md) - Learn about saving and reusing processing chains</content>
<parameter name="filePath">c:\Users\brian\Documents\GitHub\mimg\docs\installation.md