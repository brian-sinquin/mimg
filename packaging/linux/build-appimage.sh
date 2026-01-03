#!/bin/bash
set -e

# Build AppImage
# Usage: ./build-appimage.sh <binary-path> <output-dir> <version>

BINARY_PATH="${1:-../../zig-out/bin/mimg}"
OUTPUT_DIR="${2:-.}"
VERSION="${3:-1.0.0}"

echo "Building AppImage..."
echo "Binary path: $BINARY_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Version: $VERSION"

# Create AppDir structure
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

APPDIR="$TMP_DIR/mimg.AppDir"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

# Copy binary
cp "$BINARY_PATH" "$APPDIR/usr/bin/mimg"
chmod +x "$APPDIR/usr/bin/mimg"

# Create AppRun script
cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
APPDIR="$(dirname "$(readlink -f "$0")")"
exec "$APPDIR/usr/bin/mimg" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# Create .desktop file
cat > "$APPDIR/mimg.desktop" << EOF
[Desktop Entry]
Name=mimg
Exec=mimg
Icon=mimg
Type=Application
Categories=Graphics;
Comment=High-performance command-line image processing tool
Terminal=true
EOF

# Create a simple PNG icon using ImageMagick if available, otherwise create a placeholder
if command -v convert &> /dev/null; then
    convert -size 256x256 xc:none -fill blue -draw "circle 128,128 128,32" \
        -fill white -pointsize 72 -gravity center -annotate +0+0 "M" \
        "$APPDIR/mimg.png" 2>/dev/null || touch "$APPDIR/mimg.png"
else
    # Create minimal valid PNG if ImageMagick is not available
    # This is a 1x1 transparent PNG
    echo -n 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==' | base64 -d > "$APPDIR/mimg.png"
fi

cp "$APPDIR/mimg.png" "$APPDIR/.DirIcon"
cp "$APPDIR/mimg.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/mimg.png"
cp "$APPDIR/mimg.desktop" "$APPDIR/usr/share/applications/mimg.desktop"

# Download appimagetool if not available
if ! command -v appimagetool &> /dev/null; then
    echo "Downloading appimagetool..."
    APPIMAGETOOL_VERSION="13"
    APPIMAGETOOL_URL="https://github.com/AppImage/AppImageKit/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-x86_64.AppImage"
    wget -q "$APPIMAGETOOL_URL" -O "$TMP_DIR/appimagetool"
    chmod +x "$TMP_DIR/appimagetool"
    APPIMAGETOOL="$TMP_DIR/appimagetool"
else
    APPIMAGETOOL="appimagetool"
fi

# Build AppImage
OUTPUT_APPIMAGE="$OUTPUT_DIR/mimg-$VERSION-x86_64.AppImage"
ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$OUTPUT_APPIMAGE"

echo "AppImage created: $OUTPUT_APPIMAGE"
