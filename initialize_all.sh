#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed or not found in PATH." >&2
  exit 1
fi

project_paths=()

while IFS= read -r -d '' pyproject_file; do
  project_dir="$(dirname "$pyproject_file")"

  already_added=false
  if [ ${#project_paths[@]} -gt 0 ]; then
    for existing_dir in "${project_paths[@]}"; do
      if [ "$existing_dir" = "$project_dir" ]; then
        already_added=true
        break
      fi
    done
  fi

  if [ "$already_added" = false ]; then
    project_paths+=("$project_dir")
  fi
done < <(find "$repo_root" -not -path '*/.*' -type f -name "pyproject.toml" -print0)

if [ ${#project_paths[@]} -eq 0 ]; then
  echo "No pyproject.toml files found under $repo_root. Nothing to synchronize."
  exit 0
fi

for project_dir in "${project_paths[@]}"; do
  echo "==================================================================="
  echo "Syncing uv project in: $project_dir"
  echo "-------------------------------------------------------------------"
  (
    cd "$project_dir"
    if [ -d ".venv" ]; then
      echo "Removing existing .venv in $project_dir..."
      rm -rf .venv
    fi
    uv sync
  )
  echo "Completed uv sync for: $project_dir"
  echo
done

echo "All uv projects synchronized successfully."

