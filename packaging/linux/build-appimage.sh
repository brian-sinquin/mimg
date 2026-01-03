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

# Create a simple PNG icon using ImageMagick if available, otherwise create a minimal valid PNG
if command -v convert &> /dev/null; then
    convert -size 256x256 xc:none -fill blue -draw "circle 128,128 128,32" \
        -fill white -pointsize 72 -gravity center -annotate +0+0 "M" \
        "$APPDIR/mimg.png" 2>/dev/null || {
        # Fallback to minimal 1x1 transparent PNG (safe, minimal data)
        printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82' > "$APPDIR/mimg.png"
    }
else
    # Create minimal 1x1 transparent PNG (safe, minimal data)
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82' > "$APPDIR/mimg.png"
fi

cp "$APPDIR/mimg.png" "$APPDIR/.DirIcon"
cp "$APPDIR/mimg.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/mimg.png"
cp "$APPDIR/mimg.desktop" "$APPDIR/usr/share/applications/mimg.desktop"

# Download appimagetool if not available
if ! command -v appimagetool &> /dev/null; then
    echo "Downloading appimagetool..."
    APPIMAGETOOL_VERSION="13"
    APPIMAGETOOL_URL="https://github.com/AppImage/AppImageKit/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-x86_64.AppImage"
    APPIMAGETOOL_SHA256="df3baf5ca5facbecfc2f3fa6713c29ab9cefa8fd8c1eac5d283b79cab33e4acb"
    
    wget -q "$APPIMAGETOOL_URL" -O "$TMP_DIR/appimagetool"
    
    # Verify checksum if sha256sum is available
    if command -v sha256sum &> /dev/null; then
        echo "$APPIMAGETOOL_SHA256  $TMP_DIR/appimagetool" | sha256sum -c - || {
            echo "Error: Checksum verification failed for appimagetool"
            exit 1
        }
    else
        echo "Warning: sha256sum not available, skipping checksum verification"
    fi
    
    chmod +x "$TMP_DIR/appimagetool"
    APPIMAGETOOL="$TMP_DIR/appimagetool"
else
    APPIMAGETOOL="appimagetool"
fi

# Build AppImage
OUTPUT_APPIMAGE="$OUTPUT_DIR/mimg-$VERSION-x86_64.AppImage"
ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$OUTPUT_APPIMAGE"

echo "AppImage created: $OUTPUT_APPIMAGE"
