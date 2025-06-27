#!/bin/bash
path="."

# Create parent directory if it doesn't exist
mkdir -p "$path"

for dir in \
"0 - Introduction" \
"1 - Element Definition" \
"2 - Operators" \
"3 - Meshes - Connecting the Elements to Input" \
"4 - Partitioning - Parallelism and Mixed Meshes" \
"5 - Fields - Input and Output" \
"6 - Kernel Building" \
"5 - Methods - Putting it all together" \
"A - Example - Scalar Burgers Equation" \
"B - Example - Eulers Equations" \
"C - Example - Navier-Stokes Equations" \
"D - Example - Maxwells Equations"
do
    echo "$path/$dir"
    mkdir -p "$path/$dir"
done

# Optional: Verify the directories were created
echo -e "\nCreated directories:"
ls -la "$path/"
