#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="train.py"

# Directory containing your YAML configuration files
CONFIG_DIR="experiments/hp"

# Loop over each YAML file in the configuration directory
for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
    echo "Running script with configuration: $CONFIG_FILE"
    python "$PYTHON_SCRIPT" -c "$CONFIG_FILE"
    echo "Finished processing $CONFIG_FILE"
done
