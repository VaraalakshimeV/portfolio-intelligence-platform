import os

def scan_project(root="."):
    exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', 'env', 'node_modules', '.idea', '.vscode'}
    exclude_ext = {'.pyc', '.pyo', '.DS_Store'}
    
    print("=" * 60)
    print("📁 FINTECH AI PLATFORM — PROJECT FILE SCAN")
    print("=" * 60)
    
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Remove excluded dirs in-place so os.walk skips them
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        for f in filenames:
            if os.path.splitext(f)[1] in exclude_ext:
                continue
            full_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(full_path, root)
            size = os.path.getsize(full_path)
            all_files.append((rel_path, size))
    
    # Group by folder
    from collections import defaultdict
    grouped = defaultdict(list)
    for path, size in sorted(all_files):
        folder = os.path.dirname(path) or "(root)"
        grouped[folder].append((os.path.basename(path), size))
    
    total_files = 0
    for folder, files in sorted(grouped.items()):
        print(f"\n📂 {folder}/")
        for name, size in files:
            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            print(f"   ✅ {name:<40} {size_str}")
            total_files += 1
    
    print("\n" + "=" * 60)
    print(f"✅ TOTAL FILES FOUND: {total_files}")
    print("=" * 60)

if __name__ == "__main__":
    scan_project()