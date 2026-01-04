# ToonDB Python SDK Release Process

## Overview

The ToonDB Python SDK wraps pre-compiled binaries from the **main ToonDB repository** ([toondb/toondb](https://github.com/toondb/toondb)) and packages them into platform-specific Python wheels for distribution on PyPI.

## How It Works

### 1. Binary Source
- **All binaries come from** the main [toondb/toondb](https://github.com/toondb/toondb) repository
- The Python SDK does NOT compile binaries itself
- Each release pulls pre-built binaries from a specific toondb/toondb release

### 2. What Gets Packaged
Each platform-specific wheel contains:
- **Python SDK code** (`toondb` package)
- **Executables** (in `_bin/<platform>/`):
  - `toondb-bulk` - CLI tool for bulk operations
  - `toondb-server` - Standalone database server
  - `toondb-grpc-server` - gRPC server implementation
- **Native Libraries** (in `lib/<platform>/`):
  - `libtoondb_storage.*` - Storage engine FFI library
  - `libtoondb_index.*` - Indexing engine FFI library

### 3. Platform Support
The workflow builds wheels for:
- **Linux x86_64** (`manylinux_2_17_x86_64`)
- **macOS ARM64** (`macosx_11_0_arm64`) - Apple Silicon
- **Windows x64** (`win_amd64`)

### 4. Python Version Support
Wheels are compatible with:
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13

## Release Workflow

### Prerequisites
1. Ensure the desired version exists as a release in [toondb/toondb](https://github.com/toondb/toondb/releases)
2. The release must have platform-specific archives:
   - `toondb-{version}-x86_64-unknown-linux-gnu.tar.gz`
   - `toondb-{version}-aarch64-apple-darwin.tar.gz`
   - `toondb-{version}-x86_64-pc-windows-msvc.zip`

### Running a Release

1. **Go to Actions tab** in GitHub
2. **Select "Release" workflow**
3. **Click "Run workflow"**
4. **Enter parameters:**
   - `version`: The version for the Python SDK (e.g., `0.3.2`)
   - `toondb_version`: (Optional) If different from `version`, specify the toondb/toondb release to pull binaries from
   - `dry_run`: Check this to validate without publishing

### What Happens During Release

1. **Build Wheels** (parallel):
   - Downloads binaries from `toondb/toondb` release
   - Creates platform-specific wheels
   - Each wheel is self-contained with binaries for that platform

2. **Build Source Distribution**:
   - Creates `.tar.gz` with Python source code
   - Does NOT include binaries

3. **Create GitHub Release**:
   - Creates a new release tag (e.g., `v0.3.2`)
   - Attaches all wheels and source distribution
   - Generates release notes with installation instructions

4. **Publish to PyPI**:
   - Uploads all wheels and source distribution
   - Uses OIDC Trusted Publisher (no token needed)

5. **Summary**:
   - Shows comprehensive build status
   - Links to PyPI package and GitHub release
   - Shows source binary repository

## Example Release

```bash
# In toondb/toondb - first create a release there
git tag v0.3.2
git push origin v0.3.2
# Build and upload platform binaries to GitHub release

# Then in toondb-python-sdk - run the workflow
# Go to: https://github.com/toondb/toondb-python-sdk/actions
# Run workflow with:
#   version: 0.3.2
#   toondb_version: 0.3.2 (or leave blank to use same version)
#   dry_run: false
```

## Versioning Strategy

### Option 1: Same Version (Recommended)
- Python SDK version matches ToonDB core version
- Example: Both are `0.3.2`
- Simplest for users to understand

### Option 2: Independent Versions
- Python SDK has its own versioning
- Specify `toondb_version` to pull specific binaries
- Example: SDK is `1.0.0`, pulls binaries from ToonDB `0.3.2`

## Troubleshooting

### "Release not found" Error
- Ensure the toondb/toondb release exists with the correct tag format (`v0.3.2`)
- Check that platform-specific archives are attached to the release

### "No packages showing" in GitHub
- This is now fixed! The workflow creates a GitHub release with all packages attached
- Check the [Releases page](https://github.com/toondb/toondb-python-sdk/releases)

### Wheel Platform Tag Issues
- Wheels are built as `py3-none-{platform}` (platform-specific, not pure Python)
- The workflow automatically renames wheels with correct platform tags

### Missing Binaries in Wheel
- Check the workflow logs under "Copy binaries and libraries to SDK"
- Verify the toondb/toondb release has the expected binary files in the archive

## Manual Testing

To test a release locally before publishing:

```bash
# Download a wheel from the workflow artifacts or GitHub release
pip install toondb_client-0.3.2-py3-none-macosx_11_0_arm64.whl

# Verify binaries are included
python -c "import toondb; print(toondb.__file__)"
ls -la /path/to/site-packages/toondb/_bin/
ls -la /path/to/site-packages/toondb/lib/

# Test basic functionality
python -c "from toondb import Database; db = Database.open(':memory:'); print('OK')"

# Test CLI tools
toondb-bulk --help
toondb-server --help
```

## Publishing Checklist

- [ ] Verify toondb/toondb release exists with all platform binaries
- [ ] Update `version` in `pyproject.toml` if needed
- [ ] Run workflow with `dry_run: true` first
- [ ] Review workflow artifacts and logs
- [ ] Run workflow with `dry_run: false` for production
- [ ] Verify GitHub release is created
- [ ] Verify packages appear on PyPI
- [ ] Test installation: `pip install toondb-client==X.Y.Z`
- [ ] Update CHANGELOG.md with release notes

## Links

- **PyPI Package**: https://pypi.org/project/toondb-client/
- **GitHub Releases**: https://github.com/toondb/toondb-python-sdk/releases
- **Main ToonDB Repo**: https://github.com/toondb/toondb
- **ToonDB Releases**: https://github.com/toondb/toondb/releases
