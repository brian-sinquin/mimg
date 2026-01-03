# mimg Packaging Guide

This directory contains scripts and configurations for building installer packages for mimg across different platforms.

## Overview

The packaging system creates professional installers for:
- **Windows**: MSI installer using WiX Toolset
- **macOS**: DMG and PKG installers
- **Linux**: DEB, RPM, and AppImage packages

## Directory Structure

```
packaging/
├── windows/
│   ├── mimg.wxs          # WiX configuration for MSI installer
│   └── build-msi.sh      # Build script for MSI
├── macos/
│   ├── build-dmg.sh      # Build script for DMG
│   └── build-pkg.sh      # Build script for PKG
└── linux/
    ├── build-deb.sh      # Build script for DEB package
    ├── build-rpm.sh      # Build script for RPM package
    ├── build-appimage.sh # Build script for AppImage
    └── mimg.spec         # RPM spec file
```

## Building Installers Locally

### Prerequisites

#### Windows
- WiX Toolset v4 or later
  ```powershell
  dotnet tool install --global wix
  ```

#### macOS
- Xcode Command Line Tools
  ```bash
  xcode-select --install
  ```

#### Linux (Ubuntu/Debian)
- Package building tools
  ```bash
  sudo apt-get install rpm file libfuse2
  ```

### Build Commands

All build scripts follow the same pattern:
```bash
./build-<type>.sh <binary-path> <output-dir> <version>
```

#### Examples

Build Windows MSI:
```bash
cd packaging/windows
./build-msi.sh ../../zig-out/bin . 1.0.0
```

Build macOS DMG:
```bash
cd packaging/macos
./build-dmg.sh ../../zig-out/bin/mimg . 1.0.0
```

Build Linux DEB:
```bash
cd packaging/linux
./build-deb.sh ../../zig-out/bin/mimg . 1.0.0
```

Build Linux RPM:
```bash
cd packaging/linux
./build-rpm.sh ../../zig-out/bin/mimg . 1.0.0
```

Build Linux AppImage:
```bash
cd packaging/linux
./build-appimage.sh ../../zig-out/bin/mimg . 1.0.0
```

## CI/CD Integration

The packaging scripts are integrated into GitHub Actions workflows:

### Release Workflow (`new-version.yml`)
- Triggered on version tags (`v*.*.*`)
- Builds all installer types for each platform
- Uploads installers to GitHub Releases

### PR Validation Workflow (`pr-validation.yml`)
- Triggered on pull requests
- Builds all installer types for testing
- Uploads artifacts to workflow run (downloadable for 30 days)
- Allows testing installers before merge

## Installer Features

### Windows MSI
- Installs to Program Files
- Adds to Windows PATH automatically
- Creates Start Menu shortcut
- Appears in Programs and Features
- Supports clean uninstallation

### macOS DMG
- Drag-and-drop interface
- Application bundle with proper metadata
- Symbolic link to `/usr/local/bin` for easy installation
- Includes installation instructions

### macOS PKG
- Traditional macOS installer
- Installs to `/usr/local/bin`
- Automatic permissions setup
- System-wide availability

### Linux DEB
- Installs to `/usr/local/bin`
- Proper package metadata
- Integrates with APT package manager
- Clean uninstallation support

### Linux RPM
- Installs to `/usr/local/bin`
- Proper package metadata
- Integrates with YUM/DNF package managers
- Clean uninstallation support

### Linux AppImage
- Self-contained executable
- No installation required
- Works on most Linux distributions
- Includes desktop integration files

## Testing Installers

### Windows
1. Download MSI installer
2. Run installer and follow prompts
3. Verify mimg is in PATH: `mimg --version`
4. Test uninstallation from Control Panel

### macOS
For DMG:
1. Open DMG file
2. Drag mimg.app to usr-local-bin alias
3. Verify: `mimg --version`

For PKG:
1. Run PKG installer
2. Follow installation prompts
3. Verify: `mimg --version`

### Linux
For DEB:
```bash
sudo dpkg -i mimg_*_amd64.deb
mimg --version
```

For RPM:
```bash
sudo rpm -i mimg-*.rpm
mimg --version
```

For AppImage:
```bash
chmod +x mimg-*-x86_64.AppImage
./mimg-*-x86_64.AppImage --version
```

## Maintenance

### Updating Package Metadata
- Windows: Edit `packaging/windows/mimg.wxs`
- Linux DEB: Edit `packaging/linux/build-deb.sh` (control file generation)
- Linux RPM: Edit `packaging/linux/mimg.spec`
- macOS: Edit Info.plist generation in `packaging/macos/build-dmg.sh`

### Version Management
Version numbers are automatically extracted from git tags in CI/CD:
- Release builds: Use tag version (e.g., `v1.2.3` → `1.2.3`)
- PR builds: Use placeholder version `0.0.0-pr`

## Troubleshooting

### Common Issues

1. **WiX build fails on Windows**
   - Ensure WiX v4 is installed: `wix --version`
   - Check PATH includes `.dotnet\tools`

2. **DMG creation fails on macOS**
   - Ensure hdiutil is available (part of macOS)
   - Check disk space for temporary DMG files

3. **AppImage creation fails**
   - Script auto-downloads appimagetool if needed
   - Ensure `libfuse2` is installed
   - Check execute permissions on AppImage

4. **RPM build fails**
   - Ensure `rpmbuild` is available
   - Check RPM spec file syntax

## Future Enhancements

Planned improvements:
- Code signing for Windows and macOS
- Notarization for macOS
- ARM64 support for all platforms
- Homebrew formula
- Snap package for Linux
- Chocolatey package for Windows
