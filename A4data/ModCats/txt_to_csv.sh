#!/bin/bash

for file in *.txt; do
    # Skip if no .txt files exist
    [ -e "$file" ] || continue

    # Replace .txt with .csv
    mv "$file" "${file%.txt}.csv"
done

