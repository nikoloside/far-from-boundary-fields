#!/bin/bash
# Setup external method repositories for Exp3 comparison.
# Run from project root: bash experiments/exp3_external_methods/setup_repos.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPOS_DIR="$SCRIPT_DIR/repos"
mkdir -p "$REPOS_DIR"

echo "=== Setting up external method repos ==="
echo "Target: $REPOS_DIR"

# --- NDC (Neural Dual Contouring) ---
if [ ! -d "$REPOS_DIR/NDC" ]; then
    echo ""
    echo "--- Cloning NDC ---"
    git clone https://github.com/czq142857/NDC.git "$REPOS_DIR/NDC"
    echo "NDC cloned."
    echo "NOTE: NDC may require specific dependencies. Check $REPOS_DIR/NDC/README.md"
else
    echo "NDC already exists, skipping."
fi

# --- MeshUDF ---
if [ ! -d "$REPOS_DIR/MeshUDF" ]; then
    echo ""
    echo "--- Cloning MeshUDF ---"
    git clone https://github.com/cvlab-epfl/MeshUDF.git "$REPOS_DIR/MeshUDF"
    echo "MeshUDF cloned."
    echo "NOTE: MeshUDF may require specific dependencies. Check $REPOS_DIR/MeshUDF/README.md"
else
    echo "MeshUDF already exists, skipping."
fi

# --- CAP-UDF ---
if [ ! -d "$REPOS_DIR/CAP-UDF" ]; then
    echo ""
    echo "--- Cloning CAP-UDF ---"
    git clone https://github.com/junshengzhou/CAP-UDF.git "$REPOS_DIR/CAP-UDF"
    echo "CAP-UDF cloned."
    echo "NOTE: CAP-UDF may require specific dependencies. Check $REPOS_DIR/CAP-UDF/README.md"
else
    echo "CAP-UDF already exists, skipping."
fi

echo ""
echo "=== Setup complete ==="
echo "Repos at: $REPOS_DIR"
echo ""
echo "Before running exp3, you may need to install each repo's dependencies:"
echo "  - NDC:     cd $REPOS_DIR/NDC && pip install -r requirements.txt"
echo "  - MeshUDF: cd $REPOS_DIR/MeshUDF && pip install -r requirements.txt"
echo "  - CAP-UDF: cd $REPOS_DIR/CAP-UDF && pip install -r requirements.txt"
echo ""
echo "Also ensure 'igl' is installed for NDC voxelization: pip install libigl"
