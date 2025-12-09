#!/usr/bin/env bash
set -euo pipefail

echo "Attempting to auto-fix Markdown lint issues where possible."
if ! command -v markdownlint >/dev/null 2>&1; then
  echo "markdownlint not installed, installing via npm..."
  npm install -g markdownlint-cli
fi

FILES=(README.md docs/*.md docs/**/*.md *.md)
for t in "${FILES[@]}"; do
  for f in $t; do
    if [[ -f "$f" ]]; then
      echo "Auto-fixing $f"
      markdownlint --fix "$f" || true
    fi
  done
done

echo "Auto-fix complete. Rerun lint to check remaining issues."
