#!/bin/bash

# Run this script on Tursa to set up symlinks to run140's data files
# Usage: cd /path/to/experiments/multi_gpu_test && ./setup_symlinks.sh

RUN140_DIR="/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runs3B/run140"

# Create symlink to control.par
ln -sf "${RUN140_DIR}/control.par" ./control.par

# Create symlink to the CLASS HDF5 file (find it dynamically)
HDF5_FILE=$(ls ${RUN140_DIR}/class_processed_*.hdf5 2>/dev/null | head -1)
if [ -n "$HDF5_FILE" ]; then
    ln -sf "$HDF5_FILE" ./$(basename "$HDF5_FILE")
    echo "Linked: $(basename $HDF5_FILE)"
else
    echo "Warning: No class_processed_*.hdf5 file found in run140"
fi

# Also link any transfer function files if they exist
for f in ${RUN140_DIR}/transfer_function*.txt; do
    if [ -f "$f" ]; then
        ln -sf "$f" ./$(basename "$f")
        echo "Linked: $(basename $f)"
    fi
done

echo "Setup complete. Symlinks created:"
ls -la *.par *.hdf5 *.txt 2>/dev/null

