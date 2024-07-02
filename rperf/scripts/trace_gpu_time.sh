#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))
ORIGINAL_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
ORIGINAL_LOG_BREAKPOINT_LIBRARY=$LOG_BREAKPOINT_LIBRARY

LOG_BREAKPOINT_LIBRARY=$(realpath $SCRIPT_DIR/../libs/log_breakpoint/log_breakpoint.so)
CUDA_VISIBLE_DEVICES=0

export LOG_BREAKPOINT_LIBRARY
export CUDA_VISIBLE_DEVICES

if [ -z "$RPERF_OUTPUT_DIR" ]; then
  echo "RPERF_OUTPUT_DIR is not set"
  exit 1
fi

# Run the command
NSYS_OUTPUT=$RPERF_OUTPUT_DIR/gpu_time
nsys profile --stats=true --force-overwrite=true -o $NSYS_OUTPUT $@

if [ -z "$ORIGINAL_CUDA_VISIBLE_DEVICES" ]; then
  unset CUDA_VISIBLE_DEVICES
else
  export CUDA_VISIBLE_DEVICES=$ORIGINAL_CUDA_VISIBLE_DEVICES
fi

if [ -z "$ORIGINAL_LOG_BREAKPOINT_LIBRARY" ]; then
  unset LOG_BREAKPOINT_LIBRARY
else
  export LOG_BREAKPOINT_LIBRARY=$ORIGINAL_LOG_BREAKPOINT_LIBRARY
fi
