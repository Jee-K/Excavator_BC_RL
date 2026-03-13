#!/bin/bash
# Helper script to run Newton MPM examples

UV_BIN="$HOME/snap/code/220/.local/bin/uv"
cd "$(dirname "$0")"

echo "Newton MPM Examples"
echo "===================="
echo
echo "Available examples:"
echo "  1) mpm_granular        - Granular materials (sand/soil)"
echo "  2) mpm_anymal          - ANYmal robot on MPM terrain"
echo "  3) mpm_twoway_coupling - Rigid body + MPM interaction"
echo "  4) mpm_multi_material  - Multiple material types"
echo

if [ -z "$1" ]; then
    echo "Usage: $0 <example_name>"
    echo "Example: $0 mpm_granular"
    exit 1
fi

echo "Running: $1"
echo
$UV_BIN run -m newton.examples "$1" --viewer gl
