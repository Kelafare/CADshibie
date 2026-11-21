#!/usr/bin/env bash
# 本地/容器通用启动脚本。
# 用法：./run.sh --input /data/dev_final --output /results
set -euo pipefail
cd "$(dirname "$0")"
exec python infer.py "$@"
