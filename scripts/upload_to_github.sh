#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/upload_to_github.sh <github-repo-url>
# Example:
#   bash scripts/upload_to_github.sh https://github.com/USERNAME/DBSF.git

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/upload_to_github.sh <github-repo-url>"
  exit 1
fi

REPO_URL="$1"

git init
git add .
git commit -m "Initial release of DBSF watermarking framework"
git branch -M main
git remote add origin "$REPO_URL"
git push -u origin main
