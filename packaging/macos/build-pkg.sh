#!/bin/bash
set -e

# Build macOS PKG installer
# Usage: ./build-pkg.sh <binary-path> <output-dir> <version>

BINARY_PATH="${1:-../../zig-out/bin/mimg}"
OUTPUT_DIR="${2:-.}"
VERSION="${3:-1.0.0}"
IDENTIFIER="fr.brian-sinquin.mimg"

echo "Building macOS PKG installer..."
echo "Binary path: $BINARY_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Version: $VERSION"

# Create temporary directories
TMP_ROOT=$(mktemp -d)
TMP_SCRIPTS=$(mktemp -d)
trap "rm -rf $TMP_ROOT $TMP_SCRIPTS" EXIT

# Create installation directory structure
mkdir -p "$TMP_ROOT/usr/local/bin"

# Copy binary
cp "$BINARY_PATH" "$TMP_ROOT/usr/local/bin/mimg"
chmod +x "$TMP_ROOT/usr/local/bin/mimg"

# Create postinstall script to ensure proper permissions
cat > "$TMP_SCRIPTS/postinstall" << 'EOF'
#!/bin/bash
chmod +x /usr/local/bin/mimg
exit 0
EOF
chmod +x "$TMP_SCRIPTS/postinstall"

# Build the package
OUTPUT_PKG="$OUTPUT_DIR/mimg-$VERSION.pkg"

pkgbuild --root "$TMP_ROOT" \
    --identifier "$IDENTIFIER" \
    --version "$VERSION" \
    --scripts "$TMP_SCRIPTS" \
    --install-location / \
    "$OUTPUT_PKG"

echo "PKG created: $OUTPUT_PKG"
