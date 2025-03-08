#!/bin/bash
set -e

# Clone the repository into a temporary folder.
git clone https://github.com/hoyeoplee/MeLU.git

# Ensure the target directory exists.
mkdir -p TaNP/data

# Move the ml-1m folder from the clone to the target directory.
mv MeLU/movielens/ml-1m TaNP/data/

# Remove the entire cloned repository, as it's no longer needed.
rm -rf MeLU
