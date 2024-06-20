#!/bin/bash

# Determine the directory where the install.sh script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure the mfp.py script is executable
chmod +x "$SCRIPT_DIR/src/mfp.py"

# Remove the existing symbolic link, if any
sudo rm -f /usr/local/bin/mfp

# Create a new symbolic link to mfp.py in /usr/local/bin
sudo ln -s "$SCRIPT_DIR/src/mfp.py" /usr/local/bin/mfp

echo "mfp has been installed successfully and is now available in /usr/local/bin"
