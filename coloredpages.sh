#!/bin/bash
file="$1"

echo "# color pages:"
gs -o - -sDEVICE=inkcov $file | grep -v "^ 0.00000  0.00000  0.00000" | grep "^ " | wc -l

pdfinfo $file | grep Pages:

