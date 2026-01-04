#!/usr/bin/env bash
# =============================================================================
# Build Rust binary for the current platform
# =============================================================================
#
# This script is called by cibuildwheel before building the Python wheel.
# It compiles toondb-bulk and places it in the correct _bin directory.
#
# Usage:
#   ./scripts/build_rust_binary.sh
#
# Environment:
#   CARGO_BUILD_TARGET - Optional: Override target triple
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SDK_DIR="$PROJECT_DIR"
WORKSPACE_ROOT="$(dirname "$SDK_DIR")"

echo "=== ToonDB Rust Binary Build ==="
echo "Project: $PROJECT_DIR"
echo "Workspace: $WORKSPACE_ROOT"

# Detect platform and architecture
detect_platform() {
    local os arch
    
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"
    
    # Normalize OS name
    case "$os" in
        linux*)   os="linux" ;;
        darwin*)  os="darwin" ;;
        mingw*|msys*|cygwin*) os="windows" ;;
    esac
    
    # Normalize architecture
    case "$arch" in
        x86_64|amd64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        i686|i386) arch="i686" ;;
    esac
    
    echo "${os}-${arch}"
}

# Get the binary name for the platform
get_binary_name() {
    local platform="$1"
    if [[ "$platform" == windows-* ]]; then
        echo "toondb-bulk.exe"
    else
        echo "toondb-bulk"
    fi
}

# Get the Rust target triple
get_rust_target() {
    local platform="$1"
    
    case "$platform" in
        linux-x86_64)   echo "x86_64-unknown-linux-gnu" ;;
        linux-aarch64)  echo "aarch64-unknown-linux-gnu" ;;
        darwin-x86_64)  echo "x86_64-apple-darwin" ;;
        darwin-aarch64) echo "aarch64-apple-darwin" ;;
        windows-x86_64) echo "x86_64-pc-windows-msvc" ;;
        *)
            echo "Unknown platform: $platform" >&2
            exit 1
            ;;
    esac
}

# Main build logic
main() {
    local platform target binary_name bin_dir
    
    platform="${PLATFORM:-$(detect_platform)}"
    echo "Platform: $platform"
    
    binary_name="$(get_binary_name "$platform")"
    bin_dir="$SDK_DIR/src/toondb/_bin/$platform"
    
    # Create bin directory
    mkdir -p "$bin_dir"
    
    # Check if we should use a specific target
    if [[ -n "${CARGO_BUILD_TARGET:-}" ]]; then
        target="$CARGO_BUILD_TARGET"
    else
        target="$(get_rust_target "$platform")"
    fi
    
    echo "Target: $target"
    echo "Binary: $binary_name"
    echo "Output: $bin_dir/$binary_name"
    
    # Ensure Rust is available
    if ! command -v cargo &> /dev/null; then
        echo "Error: cargo not found. Install Rust first." >&2
        exit 1
    fi
    
    # Build the binary
    echo ""
    echo "Building toondb-bulk..."
    cd "$WORKSPACE_ROOT"
    
    if [[ "$target" != "$(rustc -vV | grep host | cut -d' ' -f2)" ]]; then
        # Cross-compilation: need explicit target
        cargo build --release -p toondb-tools --target "$target"
        cp "target/$target/release/$binary_name" "$bin_dir/"
    else
        # Native build
        cargo build --release -p toondb-tools
        cp "target/release/$binary_name" "$bin_dir/"
    fi
    
    # Make executable
    chmod +x "$bin_dir/$binary_name" 2>/dev/null || true
    
    echo ""
    echo "✓ Binary installed: $bin_dir/$binary_name"
    
    # Verify
    if [[ -x "$bin_dir/$binary_name" || "$platform" == windows-* ]]; then
        echo "✓ Binary is executable"
        "$bin_dir/$binary_name" --version 2>/dev/null || true
    else
        echo "Warning: Binary may not be executable"
    fi
}

main "$@"
