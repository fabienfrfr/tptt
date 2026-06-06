#!/usr/bin/env python3
"""
Code Mapper: A bidirectional synchronization tool between code and JSON.
Production-ready with strict .gitignore support via pathspec.

Usage:
python libs/code_mapper.py --to-json
python libs/code_mapper.py --from-json libs/project_structure.json
"""

import os
import json
import argparse
import pathspec

# --- Default Configurations ---
DEFAULT_OUTPUT = os.path.join("libs", "project_structure.json")
DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Utility Functions ---
def read_file_content(file_path: str) -> str:
    """Reads the content of a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str) -> None:
    """Writes a file, creating parent directories if necessary."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 File created: {file_path}")


# --- Gitignore Logic ---
def get_gitignore_spec(root_dir: str) -> pathspec.PathSpec:
    """Loads .gitignore patterns and returns a PathSpec object for strict matching."""
    patterns = [
        ".git/",
        "__pycache__/",
        "*.pyc",
        ".venv/",
        ".DS_Store",
        "node_modules/",
    ]
    gitignore_path = os.path.join(root_dir, ".gitignore")

    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            patterns.extend(f.readlines())

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


# --- JSON → Code ---
def generate_code_from_json(json_path: str) -> None:
    """Generates files from a JSON structure."""
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        project_data = json.load(f)

    for file_info in project_data["files"]:
        write_file(file_info["path"], file_info["content"])

    print("✅ Code generated successfully!")


# --- Code → JSON ---
def generate_json_from_code(root_dir: str, output_json_path: str) -> None:
    """
    Generates a JSON describing the structure of a code directory,
    respecting .gitignore strictly and excluding binary files.
    """
    files = []
    spec = get_gitignore_spec(root_dir)

    # Files to always exclude regardless of .gitignore
    internal_excludes = {".gitignore", "LICENSE", os.path.basename(output_json_path)}

    print(f"🔍 Scanning: {root_dir}")
    print(f"📁 Target: {output_json_path}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        relative_dir = os.path.relpath(dirpath, root_dir)

        # Prune ignored directories to optimize scan speed
        if relative_dir != ".":
            if spec.match_file(relative_dir):
                dirnames[:] = []
                continue

        for filename in filenames:
            if filename in internal_excludes:
                continue

            file_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(file_path, root_dir)

            if spec.match_file(relative_path):
                continue

            try:
                # Binary check: try reading as text
                with open(file_path, "tr", encoding="utf-8") as check_file:
                    check_file.read(1024)

                content = read_file_content(file_path)
                files.append({"path": relative_path, "content": content})
            except (UnicodeDecodeError, PermissionError):
                continue
            except OSError as e:
                print(f"⚠️ Error reading {relative_path}: {e}")

    project_data = {"files": files}
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(project_data, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON generated successfully with {len(files)} files.")


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Code Mapper: Sync code and JSON.")
    parser.add_argument(
        "--from-json", nargs="?", const=DEFAULT_OUTPUT, help="JSON to Code."
    )
    parser.add_argument("--to-json", nargs="*", help="Code to JSON [ROOT] [OUTPUT].")

    args = parser.parse_args()

    if args.from_json:
        generate_code_from_json(args.from_json)
    elif args.to_json is not None:
        root = args.to_json[0] if len(args.to_json) > 0 else DEFAULT_ROOT
        output = args.to_json[1] if len(args.to_json) > 1 else DEFAULT_OUTPUT
        generate_json_from_code(root, output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
