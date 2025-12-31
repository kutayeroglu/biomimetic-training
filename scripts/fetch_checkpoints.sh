#!/bin/bash

# Configuration
REMOTE_ALIAS="hpc"
REMOTE_DIR="projects/biomimetic-training"
LOCAL_DIR="/Users/kutayeroglu/projects/biomimetic-training/docs/checkpoints"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# List of files to transfer
FILES=(
    "biomimetic_checkpoint.pth"
    "anti_biomimetic_checkpoint.pth"
    "standard_checkpoint.pth"
)

echo "Connecting to alias '$REMOTE_ALIAS' to fetch checkpoints..."

# Loop through the files and transfer them
for FILE in "${FILES[@]}"; do
    echo "Syncing $FILE..."
    # rsync automatically uses your ~/.ssh/config for the 'hpc' alias
    rsync -avzP "$REMOTE_ALIAS:$REMOTE_DIR/$FILE" "$LOCAL_DIR/"
done

echo "Done! Files are in $LOCAL_DIR"