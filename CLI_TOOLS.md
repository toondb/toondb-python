# SochDB CLI Tools

> **v0.2.9** - Production-grade Python wrappers for SochDB command-line tools

After `pip install sochdb-client`, three CLI commands are globally available:

```bash
sochdb-server      # IPC server for multi-process access
sochdb-bulk        # High-performance vector operations
sochdb-grpc-server # gRPC server for remote vector search
```

---

## sochdb-server

Multi-process database access via Unix domain sockets.

### Quick Start

```bash
# Start server
sochdb-server --db ./my_database

# Check status
sochdb-server status --db ./my_database

# Stop server
sochdb-server stop --db ./my_database
```

### Features

| Feature | Description |
|---------|-------------|
| **Stale socket detection** | Automatically cleans up orphaned socket files |
| **Health checks** | Waits for server to be ready before returning |
| **Graceful shutdown** | Handles SIGTERM/SIGINT for clean teardown |
| **PID tracking** | Writes PID file for process management |
| **Permission checks** | Validates directory is writable before starting |
| **Actionable errors** | Clear error messages with fix suggestions |

### Options

```
Usage: sochdb-server [OPTIONS] [COMMAND]

Commands:
  stop     Stop a running server
  status   Check server status

Options:
  -d, --db PATH           Database directory [default: ./sochdb_data]
  -s, --socket PATH       Unix socket path [default: <db>/sochdb.sock]
      --max-clients N     Maximum connections [default: 100]
      --timeout-ms MS     Connection timeout [default: 30000]
      --log-level LEVEL   trace/debug/info/warn/error [default: info]
      --no-wait           Don't wait for server to be ready
      --version           Show version and exit
```

### Examples

```bash
# Development
sochdb-server --db ./dev_db --log-level debug

# Production
sochdb-server \
    --db /var/lib/sochdb/production \
    --socket /var/run/sochdb.sock \
    --max-clients 500 \
    --timeout-ms 60000 \
    --log-level info

# Check if running
sochdb-server status --db ./my_database
# Output: [Server] Running (PID: 12345)

# Graceful stop
sochdb-server stop --db ./my_database
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SOCHDB_SERVER_PATH` | Override bundled binary path |

### Error Handling

```bash
# Socket already in use
$ sochdb-server --db ./my_db
[Server] Error: Socket already in use (PID: 12345): ./my_db/sochdb.sock
         Another sochdb-server instance may be running.
         Use 'sochdb-server stop --socket ./my_db/sochdb.sock' to stop it.

# Permission denied
$ sochdb-server --db /root/db
[Server] Error: Database directory is not writable: /root/db

# Binary not found
$ sochdb-server --db ./my_db
[Server] Error: sochdb-server binary not found.
Searched:
  - SOCHDB_SERVER_PATH environment variable
  - Bundled in package (_bin/)
  - System PATH

To fix:
  1. Reinstall: pip install --force-reinstall sochdb-client
  2. Or build: cargo build --release -p sochdb-server
  3. Or set: export SOCHDB_SERVER_PATH=/path/to/sochdb-server
```

---

## sochdb-bulk

High-performance vector index building and querying.

### Quick Start

```bash
# Build HNSW index
sochdb-bulk build-index \
    --input embeddings.npy \
    --output index.hnsw \
    --dimension 768

# Query index
sochdb-bulk query \
    --index index.hnsw \
    --query query.raw \
    --k 10

# Get index info
sochdb-bulk info --index index.hnsw
```

### Features

| Feature | Description |
|---------|-------------|
| **Input validation** | Checks file exists, readable, correct extension |
| **Output validation** | Checks directory writable, handles overwrites |
| **Progress reporting** | Shows file sizes and operation progress |
| **Error recovery** | Actionable error messages with suggestions |

### Commands

#### build-index

Build an HNSW vector index from embeddings.

```bash
sochdb-bulk build-index \
    --input vectors.npy \        # .npy or .raw format
    --output index.hnsw \        # Output index path
    --dimension 768 \            # Vector dimension
    --max-connections 16 \       # HNSW M parameter
    --ef-construction 100 \      # Build-time ef
    --threads 0 \                # 0 = auto-detect
    --batch-size 1000 \          # Vectors per batch
    --metric cosine \            # cosine/l2/ip
    --overwrite                  # Overwrite existing
```

#### query

Query an HNSW index for nearest neighbors.

```bash
sochdb-bulk query \
    --index index.hnsw \     # Index file
    --query query.raw \      # Query vector (.raw or .npy)
    --k 10 \                 # Number of neighbors
    --ef 64                  # Search ef parameter
```

#### info

Display index metadata.

```bash
sochdb-bulk info --index index.hnsw

# Output:
# Dimension: 768
# Vectors: 100000
# Max connections: 16
# Entry point: 42
```

#### convert

Convert between vector formats.

```bash
sochdb-bulk convert \
    --input vectors.npy \
    --output vectors.raw \
    --to-format raw_f32 \
    --dimension 768
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SOCHDB_BULK_PATH` | Override bundled binary path |

### Error Handling

```bash
# Input not found
$ sochdb-bulk build-index --input missing.npy --output out.hnsw --dimension 768
[Bulk] Error: Input file not found: /path/to/missing.npy

# Output exists
$ sochdb-bulk build-index --input data.npy --output existing.hnsw --dimension 768
[Bulk] Error: Output file already exists: /path/to/existing.hnsw
       Use --overwrite to replace it

# Invalid extension
$ sochdb-bulk build-index --input data.txt --output out.hnsw --dimension 768
[Bulk] Error: Invalid file extension: .txt
       Expected one of: .npy, .raw, .bin
```

---

## sochdb-grpc-server

gRPC server for remote vector search operations.

### Quick Start

```bash
# Start server
sochdb-grpc-server

# Custom host and port
sochdb-grpc-server --host 0.0.0.0 --port 50051

# Check status
sochdb-grpc-server status --port 50051
```

### Features

| Feature | Description |
|---------|-------------|
| **Port checking** | Verifies port is available before starting |
| **Process detection** | Identifies what process is using a port |
| **Privileged port check** | Warns about ports < 1024 requiring root |
| **Health checks** | Waits for gRPC endpoint to be ready |
| **Graceful shutdown** | Handles signals for clean teardown |

### Options

```
Usage: sochdb-grpc-server [OPTIONS] [COMMAND]

Commands:
  status   Check if server is running

Options:
      --host HOST    Bind address [default: 127.0.0.1]
  -p, --port PORT    Listen port [default: 50051]
  -d, --debug        Enable debug logging
      --no-wait      Don't wait for server to be ready
      --version      Show version and exit
```

### Examples

```bash
# Local development
sochdb-grpc-server --debug

# Production (all interfaces)
sochdb-grpc-server --host 0.0.0.0 --port 50051

# Check if running
sochdb-grpc-server status --port 50051
# Output: [gRPC] Running on 127.0.0.1:50051
```

### Python Client Usage

```python
import grpc
from sochdb_pb2 import VectorSearchRequest
from sochdb_pb2_grpc import VectorServiceStub

# Connect
channel = grpc.insecure_channel('localhost:50051')
stub = VectorServiceStub(channel)

# Search
request = VectorSearchRequest(
    index_path="index.hnsw",
    query_vector=query_vec.tobytes(),
    k=10,
    ef_search=64
)
response = stub.Search(request)

for neighbor in response.neighbors:
    print(f"ID: {neighbor.id}, Distance: {neighbor.distance}")
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SOCHDB_GRPC_SERVER_PATH` | Override bundled binary path |

### Error Handling

```bash
# Port in use
$ sochdb-grpc-server --port 8080
[gRPC] Error: Port 8080 is already in use by nginx (PID: 1234)
       Try a different port with --port <PORT>

# Privileged port
$ sochdb-grpc-server --port 80
[gRPC] Error: Port 80 requires root privileges
       Use a port >= 1024 or run with sudo
```

---

## Exit Codes

All CLI tools use consistent exit codes:

| Code | Name | Description |
|------|------|-------------|
| 0 | SUCCESS | Operation completed successfully |
| 1 | GENERAL_ERROR | General error |
| 2 | BINARY_NOT_FOUND | Native binary not found |
| 3 | PORT/SOCKET_IN_USE | Port or socket already in use |
| 4 | PERMISSION_DENIED | Permission denied |
| 5 | STARTUP_FAILED | Server failed to start |
| 130 | INTERRUPTED | Interrupted by SIGINT (Ctrl+C) |

---

## Comparison: Before vs After

### Before (v0.2.7)

```bash
# Had to find and use absolute paths
/path/to/venv/lib/python3.11/site-packages/sochdb/_bin/macos/sochdb-server --db ./my_db

# No status checking
ps aux | grep sochdb-server

# Manual cleanup of stale sockets
rm ./my_db/sochdb.sock

# No validation
./sochdb-bulk build-index --input missing.npy  # Cryptic error
```

### After (v0.2.9)

```bash
# Simple, global commands
sochdb-server --db ./my_db

# Built-in status checking
sochdb-server status --db ./my_db

# Automatic stale socket cleanup
sochdb-server --db ./my_db  # Cleans up stale sockets automatically

# Clear, actionable errors
sochdb-bulk build-index --input missing.npy
# [Bulk] Error: Input file not found: /path/to/missing.npy
```

---

## Troubleshooting

### Binary Not Found

```bash
# Check binary resolution
python -c "from sochdb.cli_server import get_server_binary; print(get_server_binary())"

# Set path manually
export SOCHDB_SERVER_PATH=/path/to/sochdb-server
export SOCHDB_BULK_PATH=/path/to/sochdb-bulk
export SOCHDB_GRPC_SERVER_PATH=/path/to/sochdb-grpc-server
```

### Permission Issues

```bash
# Make binary executable
chmod +x /path/to/_bin/*/sochdb-*

# Check directory permissions
ls -la ./my_database
```

### Port Already in Use

```bash
# Find process using port
lsof -i :50051

# Kill process
kill <PID>
```

---

*Last updated: January 2026 (v0.2.9)*
