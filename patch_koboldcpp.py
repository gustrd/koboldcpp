#!/usr/bin/env python3
"""
Script to patch koboldcpp.py with auto-banned tokens
Usage: python patch_koboldcpp.py [token1 token2 token3 ...]
If no tokens provided, uses defaults: <args> </args> <parameters> </parameters> <arguments> </arguments>
"""

import sys
import shutil
import re

# Default tokens to ban
DEFAULT_TOKENS = ["<args>", "</args>", "<parameters>", "</parameters>", "<arguments>", "</arguments>"]

def main():
    # Use provided tokens or defaults
    if len(sys.argv) > 1:
        tokens = sys.argv[1:]
        print(f"Using custom tokens: {tokens}")
    else:
        tokens = DEFAULT_TOKENS
        print(f"No tokens provided, using defaults: {tokens}")

    # Build Python list string
    python_list = str(tokens)
    print(f"Auto-banned tokens list: {python_list}")

    # Check if source file exists
    try:
        with open('koboldcpp.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("Error: koboldcpp.py not found in current directory")
        sys.exit(1)

    # Create patched copy
    print("Copying koboldcpp.py to koboldcpp_patched.py...")
    shutil.copy2('koboldcpp.py', 'koboldcpp_patched.py')

    # Apply the patch
    print("Applying patch...")

    # Find the line with banned_tokens = genparams.get and add auto-banned tokens after it
    pattern = r"(    banned_tokens = genparams\.get\('banned_tokens', banned_strings\))"
    replacement = f"""\\1
    # Auto-ban specific tokens to prevent unwanted XML-like tags
    auto_banned = {python_list}
    banned_tokens = list(banned_tokens) + auto_banned if banned_tokens else auto_banned"""

    with open('koboldcpp_patched.py', 'r', encoding='utf-8') as f:
        patched_content = f.read()

    # Apply the replacement
    patched_content = re.sub(pattern, replacement, patched_content)

    # Write the patched content
    with open('koboldcpp_patched.py', 'w', encoding='utf-8') as f:
        f.write(patched_content)

    print("✓ Patch applied successfully!")
    print("✓ Created: koboldcpp_patched.py")
    print("")
    print("The following tokens will be automatically banned in all generation requests:")
    for token in tokens:
        print(f"  - {token}")

if __name__ == "__main__":
    main()
