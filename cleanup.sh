#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <experiment_directory>"
  exit 1
fi

EXPERIMENT_DIR="$1"

# Loop over trial directories
find "$EXPERIMENT_DIR" -type d -name "trial_*" | while read -r trial_dir; do
  echo "Processing trial: $trial_dir"

  env_dir=$(find "$trial_dir" -type d -name "*Env*" | head -n 1)
  if [ -z "$env_dir" ]; then
    echo "  No Env dir found"
    continue
  fi
  echo "  Found Env dir: $env_dir"

  # Call the cleanup script for this Env directory
  ./cleanup_checkpoints.sh "$env_dir"
done
