#!/bin/bash

# Check input
if [ -z "$1" ]; then
  echo "Usage: $0 <env_directory>"
  exit 1
fi

ENV_DIR="$1"

# Traverse algorithm dirs
for algo_dir in "$ENV_DIR"/*/; do
  [ -d "$algo_dir" ] || continue
  echo "  Algorithm: $algo_dir"

  # Traverse candidate dirs
  for candidate_dir in "$algo_dir"/*/; do
    [ -d "$candidate_dir" ] || continue
    echo "    Candidate: $candidate_dir"

    summary_file="$candidate_dir/run_summary.json"
    if [ ! -f "$summary_file" ]; then
      echo "      No run_summary.json"
      continue
    fi

    # Extract best epoch
    best_epoch=$(jq -r '."best_epoch-best_epoch"' "$summary_file")
    if [ -z "$best_epoch" ] || [ "$best_epoch" == "null" ]; then
      echo "      Invalid best_epoch"
      continue
    fi
    echo "      Best epoch: $best_epoch"

    # Remove other checkpoints
    for ckpt_file in "$candidate_dir"/checkpoint_epoch*.pth; do
      [[ "$ckpt_file" == *"checkpoint_epoch${best_epoch}.pth" ]] && continue
      echo "      Removing: $ckpt_file"
      rm -f "$ckpt_file"
    done
  done
done
