#!/bin/bash

# Script to patch koboldcpp.py with auto-banned tokens
# Usage: ./patch_koboldcpp.sh [token1 token2 token3 ...]
# If no tokens provided, uses defaults: <args> </args> <parameters> </parameters> <arguments> </arguments>

# Default tokens to ban
DEFAULT_TOKENS=("<args>" "</args>" "<parameters>" "</parameters>" "<arguments>" "</arguments>")

# Use provided tokens or defaults
if [ $# -eq 0 ]; then
    TOKENS=("${DEFAULT_TOKENS[@]}")
    echo "No tokens provided, using defaults: ${TOKENS[@]}"
else
    TOKENS=("$@")
    echo "Using custom tokens: ${TOKENS[@]}"
fi

# Build Python list string
PYTHON_LIST="["
for i in "${!TOKENS[@]}"; do
    if [ $i -gt 0 ]; then
        PYTHON_LIST+=", "
    fi
    PYTHON_LIST+="'${TOKENS[$i]}'"
done
PYTHON_LIST+="]"

echo "Auto-banned tokens list: $PYTHON_LIST"

# Check if source file exists
if [ ! -f "koboldcpp.py" ]; then
    echo "Error: koboldcpp.py not found in current directory"
    exit 1
fi

# Create patched copy
echo "Copying koboldcpp.py to koboldcpp_patched.py..."
cp koboldcpp.py koboldcpp_patched.py

# Create the patch
# Find the line with 'banned_tokens = genparams.get' and add auto-banned tokens after it
echo "Applying patch..."

# Use sed to insert the auto-ban code after the banned_tokens line
sed -i '/banned_tokens = genparams\.get.*banned_strings)/a\    # Auto-ban specific tokens to prevent unwanted XML-like tags\n    auto_banned = '"$PYTHON_LIST"'\n    banned_tokens = list(banned_tokens) + auto_banned if banned_tokens else auto_banned' koboldcpp_patched.py

if [ $? -eq 0 ]; then
    echo "✓ Patch applied successfully!"
    echo "✓ Created: koboldcpp_patched.py"
    echo ""
    echo "The following tokens will be automatically banned in all generation requests:"
    for token in "${TOKENS[@]}"; do
        echo "  - $token"
    done
else
    echo "✗ Error applying patch"
    exit 1
fi
