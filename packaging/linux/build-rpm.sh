#!/bin/bash
set -e

# Build RPM package
# Usage: ./build-rpm.sh <binary-path> <output-dir> <version>

BINARY_PATH="${1:-../../zig-out/bin/mimg}"
OUTPUT_DIR="${2:-.}"
VERSION="${3:-1.0.0}"

echo "Building RPM package..."
echo "Binary path: $BINARY_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Version: $VERSION"

# Create RPM build environment
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

mkdir -p "$TMP_DIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Copy binary to SOURCES
cp "$BINARY_PATH" "$TMP_DIR/SOURCES/mimg"

# Copy and process spec file
sed "s/%{version}/$VERSION/g" mimg.spec > "$TMP_DIR/SPECS/mimg.spec"

# Build the RPM
rpmbuild --define "_topdir $TMP_DIR" \
    --define "version $VERSION" \
    -bb "$TMP_DIR/SPECS/mimg.spec"

# Find and copy the generated RPM
find "$TMP_DIR/RPMS" -name "*.rpm" -exec cp {} "$OUTPUT_DIR/" \;

RPM_FILE=$(find "$TMP_DIR/RPMS" -name "*.rpm" -exec basename {} \;)
echo "RPM package created: $OUTPUT_DIR/$RPM_FILE"
