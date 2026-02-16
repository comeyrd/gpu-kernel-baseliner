#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <executable_with_args>"
    exit 1
fi

cleanup() {
    echo "Restoring GPU clocks..."
    sudo nvidia-smi --reset-gpu-clocks
}

trap cleanup EXIT

echo "Locking GPU clocks to TDP..."
sudo nvidia-smi --lock-gpu-clocks=tdp,tdp

echo "Executing: $@"
"$@"
