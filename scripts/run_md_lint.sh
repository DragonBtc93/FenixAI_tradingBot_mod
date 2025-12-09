#!/usr/bin/env bash
set -euo pipefail

echo "Running markdownlint across docs and root markdown files..."
if ! command -v markdownlint >/dev/null 2>&1; then
  echo "markdownlint not installed, installing via npm..."
  npm install -g markdownlint-cli
fi

TARGETS=(README.md docs/*.md docs/**/*.md *.md)
for t in "${TARGETS[@]}"; do
  for f in $t; do
    if [[ -f "$f" ]]; then
      echo "Linting $f"
      markdownlint "$f" || true
    fi
  done
done

echo "Markdown linting complete (errors reported above)."
