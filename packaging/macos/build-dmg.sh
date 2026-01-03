#!/bin/bash
set -e

# Build macOS DMG installer
# Usage: ./build-dmg.sh <binary-path> <output-dir> <version>

BINARY_PATH="${1:-../../zig-out/bin/mimg}"
OUTPUT_DIR="${2:-.}"
VERSION="${3:-1.0.0}"
APP_NAME="mimg"

echo "Building macOS DMG installer..."
echo "Binary path: $BINARY_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Version: $VERSION"

# Create temporary directory for DMG contents
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Create application bundle structure
APP_BUNDLE="$TMP_DIR/$APP_NAME.app"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# Copy binary
cp "$BINARY_PATH" "$APP_BUNDLE/Contents/MacOS/$APP_NAME"
chmod +x "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

# Create Info.plist
cat > "$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>fr.brian-sinquin.mimg</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
</dict>
</plist>
EOF

# Create symbolic link to /usr/local/bin for easy installation
ln -s /usr/local/bin "$TMP_DIR/usr-local-bin"

# Create README for installation
cat > "$TMP_DIR/README.txt" << EOF
mimg - Command-line Image Processing Tool

Installation Instructions:
1. Drag the mimg.app to the usr-local-bin alias to install
2. Open Terminal and type 'mimg' to verify installation

Alternatively, you can manually copy the binary:
   sudo cp mimg.app/Contents/MacOS/mimg /usr/local/bin/

For more information, visit:
https://github.com/brian-sinquin/mimg
EOF

# Create DMG
OUTPUT_DMG="$OUTPUT_DIR/mimg-$VERSION.dmg"

# Use hdiutil to create DMG
hdiutil create -volname "mimg" \
    -srcfolder "$TMP_DIR" \
    -ov -format UDZO \
    "$OUTPUT_DMG"

echo "DMG created: $OUTPUT_DMG"
