#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ===========================
# Configuration Variables
# ===========================

# Absolute path to the SuperGans project directory
PROJECT_DIR="/Users/Lerberber/Desktop/Bhat/SuperGans"

# Name of the virtual environment directory
VENV_DIR="supergan_env"

# Configuration files
MODEL_CONF="model.conf"
CONFIG_FILE="config.toml"

# Dataset and classifier paths
DATASET_PATH="data/LSTM_accelerometer.h5"
CLASSIFIER_PATH="data/LSTM_accelerometer.h5"  # Assuming classifier is in the same .h5 file
CLASS_LABEL=0  # Replace with your desired class label

# Python executable
PYTHON_EXEC="python3"

# ===========================
# Functions
# ===========================

# Function to print messages
print_message() {
    echo "========================================"
    echo "$1"
    echo "========================================"
}

# ===========================
# Script Execution
# ===========================

# Step 1: Navigate to the project directory
print_message "Navigating to the project directory..."
cd "$PROJECT_DIR"

# Step 2: Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    print_message "Creating virtual environment..."
    $PYTHON_EXEC -m venv "$VENV_DIR"
else
    print_message "Virtual environment already exists."
fi

# Step 3: Activate the virtual environment
print_message "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Step 4: Upgrade pip
print_message "Upgrading pip..."
pip install --upgrade pip

# Step 5: Install dependencies
print_message "Installing dependencies..."
# Check if setup.py exists and install via it
if [ -f "setup.py" ]; then
    pip install -e .
else
    # Fallback to package_list.txt
    if [ -f "package_list.txt" ]; then
        pip install -r package_list.txt
    else
        echo "Error: Neither setup.py nor package_list.txt found!"
        deactivate
        exit 1
    fi
fi

# Step 6: Create config.toml if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    print_message "Creating $CONFIG_FILE..."
    cat <<EOL > "$CONFIG_FILE"
data_file_path = "$DATASET_PATH"
classifier_path = "$CLASSIFIER_PATH"
class_label = $CLASS_LABEL
EOL
else
    print_message "$CONFIG_FILE already exists. Skipping creation."
fi

# Step 7: Verify model.conf exists
if [ ! -f "$MODEL_CONF" ]; then
    echo "Error: $MODEL_CONF not found in the project directory!"
    deactivate
    exit 1
fi

# Step 8: Run the main script
print_message "Running SuperGAN..."
python main.py "$CONFIG_FILE"

# Step 9: Deactivate the virtual environment
print_message "Deactivating virtual environment..."
deactivate

print_message "SuperGAN has been executed successfully."

