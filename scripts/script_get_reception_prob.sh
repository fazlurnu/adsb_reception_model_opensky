#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Directory containing folders
directory="$1"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory $directory does not exist."
    exit 1
fi

# Iterate through each folder in the directory
for folder in "$directory"/*; do
    if [ -d "$folder" ]; then
        echo "Running script for folder: $folder"
        python get_reception_prob.py "$folder"
    fi
done
