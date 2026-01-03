#!/bin/bash
set -e

# Build Windows MSI installer using WiX Toolset
# Usage: ./build-msi.sh <binary-path> <output-dir> <version>

BINARY_PATH="${1:-../../zig-out/bin}"
OUTPUT_DIR="${2:-.}"
VERSION="${3:-1.0.0}"

echo "Building Windows MSI installer..."
echo "Binary path: $BINARY_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Version: $VERSION"

# Ensure WiX is available
if ! command -v wix &> /dev/null; then
    echo "WiX toolset not found. Please install WiX Toolset v3 or v4."
    exit 1
fi

# Clean previous build
rm -f mimg.wixobj mimg.wixpdb

# Build the MSI
wix build mimg.wxs \
    -d BinaryPath="$BINARY_PATH" \
    -out "$OUTPUT_DIR/mimg-installer.msi"

echo "MSI installer created: $OUTPUT_DIR/mimg-installer.msi"
