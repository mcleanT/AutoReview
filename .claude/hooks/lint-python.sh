#!/usr/bin/env bash
# Auto-lint Python files after Edit/Write tool use
set -euo pipefail

input="$(cat)"
file="$(echo "$input" | jq -r '.tool_input.file_path // empty')"

# Only process Python files
if [[ -n "$file" && "$file" == *.py && -f "$file" ]]; then
  ruff check --fix "$file" 2>/dev/null || true
  ruff format "$file" 2>/dev/null || true
fi

exit 0
