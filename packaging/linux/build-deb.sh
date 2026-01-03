#!/bin/bash
set -e

# Build Debian package
# Usage: ./build-deb.sh <binary-path> <output-dir> <version>

BINARY_PATH="${1:-../../zig-out/bin/mimg}"
OUTPUT_DIR="${2:-.}"
VERSION="${3:-1.0.0}"

echo "Building Debian package..."
echo "Binary path: $BINARY_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Version: $VERSION"

# Create temporary directory for package structure
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

PKG_DIR="$TMP_DIR/mimg_${VERSION}_amd64"
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/usr/local/bin"

# Copy binary
cp "$BINARY_PATH" "$PKG_DIR/usr/local/bin/mimg"
chmod +x "$PKG_DIR/usr/local/bin/mimg"

# Create control file
cat > "$PKG_DIR/DEBIAN/control" << EOF
Package: mimg
Version: $VERSION
Section: graphics
Priority: optional
Architecture: amd64
Maintainer: Brian Sinquin <brian.sinquin@example.com>
Description: High-performance command-line image processing tool
 mimg is a fast image processing tool written in Zig that supports
 multiple formats (PNG, JPEG, BMP, QOI) with chainable modifiers
 for complex transformations.
 .
 Features:
  - Fast image processing
  - Multiple format support
  - Chainable modifiers
  - Built-in presets
  - Zero external dependencies
Homepage: https://github.com/brian-sinquin/mimg
EOF

# Build the package
dpkg-deb --build "$PKG_DIR"

# Move to output directory
mv "$PKG_DIR.deb" "$OUTPUT_DIR/"

echo "DEB package created: $OUTPUT_DIR/mimg_${VERSION}_amd64.deb"
