#!/usr/bin/env bash
# Verify v3.4 frozen artifact integrity
cd "$(dirname "$0")"
shasum -a 256 -c SHA256SUMS
