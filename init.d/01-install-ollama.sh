#!/usr/bin/env bash
mkdir -p "$RENKU_MOUNT_DIR/ollama"
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar x --zstd -C "$RENKU_MOUNT_DIR/ollama"
echo 'PATH="$PATH:$RENKU_MOUNT_DIR/ollama/bin"' >>~/.bashrc
