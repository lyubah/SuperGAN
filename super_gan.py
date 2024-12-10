import os
import subprocess
from pathlib import Path

# ===========================
# Configuration Variables
# ===========================

# Base directory for the project
BASE_DIR = Path("/Users/Lerberber/Desktop/Bhat/SuperGans")

# Name of the virtual environment directory
VENV_DIR = BASE_DIR / "supergan_env"

# Paths to configuration files
MODEL_CONF = BASE_DIR / "model.conf"
CONFIG_FILE = BASE_DIR / "config.toml"

# Dataset and classifier paths
DATASET_PATH = BASE_DIR / "data/LSTM_accelerometer.h5"
CLASSIFIER_PATH = BASE_DIR / "LSTM_accelerometer.h5"

# Class label to generate
CLASS_LABEL = 0

# Python executable
PYTHON_EXEC = "python3"

# ===========================
# Helper Functions
# ===========================

def print_message(message):
    """Prints a formatted message."""
    print("=" * 40)
    print(message)
    print("=" * 40)

def create_virtual_environment():
    """Creates a virtual environment if it doesn't exist."""
    if not VENV_DIR.exists():
        print_message("Creating virtual environment...")
        subprocess.run([PYTHON_EXEC, "-m", "venv", str(VENV_DIR)], check=True)
    else:
        print_message("Virtual environment already exists.")

def activate_virtual_environment():
    """Activates the virtual environment."""
    activate_script = VENV_DIR / "bin" / "activate"
    if not activate_script.exists():
        raise FileNotFoundError(f"Virtual environment not found at {activate_script}. Please create it first.")
    return str(activate_script)

def install_dependencies():
    """Installs dependencies using setup.py or package_list.txt."""
    print_message("Installing dependencies...")
    if (BASE_DIR / "setup.py").exists():
        subprocess.run([PYTHON_EXEC, "setup.py", "install"], cwd=BASE_DIR, check=True)
    elif (BASE_DIR / "package_list.txt").exists():
        subprocess.run(["pip", "install", "-r", "package_list.txt"], cwd=BASE_DIR, check=True)
    else:
        raise FileNotFoundError("Neither setup.py nor package_list.txt found for dependency installation.")

def create_config_toml():
    """Creates the config.toml file."""
    if not CONFIG_FILE.exists():
        print_message(f"Creating {CONFIG_FILE}...")
        with open(CONFIG_FILE, "w") as file:
            file.write(f"""data_file_path = "{DATASET_PATH}"
classifier_path = "{CLASSIFIER_PATH}"
class_label = {CLASS_LABEL}
""")
    else:
        print_message(f"{CONFIG_FILE} already exists. Skipping creation.")

def verify_model_conf():
    """Verifies that model.conf exists."""
    if not MODEL_CONF.exists():
        raise FileNotFoundError(f"{MODEL_CONF} not found in the project directory.")

def run_supergan():
    """Runs the main SuperGAN script."""
    print_message("Running SuperGAN...")
    subprocess.run([PYTHON_EXEC, "main.py", str(CONFIG_FILE)], cwd=BASE_DIR, check=True)

# ===========================
# Script Execution
# ===========================

def main():
    """Main function to set up and run SuperGAN."""
    try:
        os.chdir(BASE_DIR)  # Navigate to the project directory
        create_virtual_environment()  # Step 1: Create virtual environment
        activate_script = activate_virtual_environment()  # Step 2: Activate virtual environment
        subprocess.run(["source", activate_script], shell=True)  # Activate the virtual environment
        subprocess.run(["pip", "install", "--upgrade", "pip"], check=True)  # Upgrade pip
        install_dependencies()  # Step 3: Install dependencies
        create_config_toml()  # Step 4: Create config.toml
        verify_model_conf()  # Step 5: Verify model.conf
        run_supergan()  # Step 6: Run SuperGAN
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print_message("SuperGAN execution completed.")

if __name__ == "__main__":
    main()

