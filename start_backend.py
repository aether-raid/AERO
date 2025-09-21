#!/usr/bin/env python3
"""
ML Researcher LangGraph - Backend Startup Script
===============================================

Simple script to start the FastAPI backend server.
"""

import uvicorn
import sys
import os
import glob
import hashlib
import subprocess
from pathlib import Path

def _hash_contents(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _collect_and_combine_requirements(project_root: Path) -> Path:
    """Combine all requirements*.txt files in root into a temp combined file.

    - Skips comments/blank lines
    - Deduplicates exact lines while preserving first occurrence order
    - Returns path to combined file (created in project root as .combined_requirements.txt)
    - If nothing to combine, returns an empty file
    """
    patterns = ["requirements.txt", "requirements_*.txt"]
    files = []
    for pattern in patterns:
        files.extend(sorted(project_root.glob(pattern)))
    # Keep only existing regular files
    files = [f for f in files if f.is_file()]
    seen = set()
    ordered = []
    for f in files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                raw = line.rstrip()
                if not raw or raw.lstrip().startswith("#"):
                    continue
                if raw not in seen:
                    seen.add(raw)
                    ordered.append(raw)
        except Exception as e:
            print(f"⚠️  Could not read {f.name}: {e}")
    combined_path = project_root / ".combined_requirements.txt"
    combined_text = "\n".join(ordered) + ("\n" if ordered else "")
    combined_path.write_text(combined_text, encoding="utf-8")
    return combined_path


def _should_install(all_req_file: Path, cache_file: Path) -> bool:
    if os.getenv("SKIP_AUTO_INSTALL") == "1":
        print("⏭️  SKIP_AUTO_INSTALL=1 set: skipping dependency installation phase")
        return False
    if not all_req_file.exists():
        return False
    contents = all_req_file.read_text(encoding="utf-8")
    current_sig = _hash_contents(contents)
    if cache_file.exists():
        prev = cache_file.read_text(encoding="utf-8").strip()
        if prev == current_sig:
            print("✅ Dependencies appear up-to-date (signature match)")
            return False
    # Basic sentinel import check to avoid reinstall loop if signature changed only by ordering
    return True


def _install_requirements(all_req_file: Path, cache_file: Path):
    if not all_req_file.exists() or all_req_file.stat().st_size == 0:
        print("ℹ️  No combined requirements to install.")
        return
    print("📦 Installing/Updating dependencies from all requirement files...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(all_req_file)]
    verbose = os.getenv("REQUIREMENTS_VERBOSE") == "1"
    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, check=False)
        if verbose:
            print("(pip output above)")
        if result.returncode != 0:
            print("❌ pip install reported errors (continuing to try startup)")
            if not verbose:
                print(result.stderr[-2000:])  # tail of stderr
        else:
            sig = _hash_contents(all_req_file.read_text(encoding="utf-8"))
            cache_file.write_text(sig, encoding="utf-8")
            print("✅ Dependencies installed / verified")
    except Exception as e:
        print(f"⚠️  Dependency installation failed unexpectedly: {e}")


def main():
    print("🚀 Starting ML Researcher LangGraph Backend...")
    print("=" * 50)

    project_root = Path(__file__).resolve().parent

    # 1. Combine all requirement files
    combined_requirements = _collect_and_combine_requirements(project_root)
    cache_sig_file = project_root / ".combined_requirements.sig"

    # 2. Optionally install (skip if signature unchanged or env var set)
    if _should_install(combined_requirements, cache_sig_file):
        _install_requirements(combined_requirements, cache_sig_file)
    else:
        print("ℹ️  Skipping installation step")

    # 3. Core dependency sanity check
    missing = []
    try:
        import fastapi  # noqa: F401
    except ImportError:
        missing.append("fastapi")
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        missing.append("uvicorn")
    # torch / sentence-transformers are optional for semantic features, but note if missing
    opt_missing = []
    try:
        import sentence_transformers  # noqa: F401
    except (ImportError, Exception) as e:
        print(f"⚠️  sentence-transformers import failed: {e}")
        opt_missing.append("sentence-transformers")
    try:
        import torch  # noqa: F401
    except (ImportError, Exception) as e:
        print(f"⚠️  torch import failed: {e}")
        opt_missing.append("torch")

    if missing:
        print(f"❌ Critical dependencies missing: {', '.join(missing)}")
        print("   Attempt manual install: pip install -r requirements.txt")
        sys.exit(1)

    if opt_missing:
        print(f"⚠️  Optional components not available: {', '.join(opt_missing)} (semantic features may be limited)")

    # 4. Import application (this triggers model/workflow initialization)
    try:
        import ml_researcher_langgraph  # noqa: F401
    except ImportError as e:
        print(f"❌ Failed to import application module: {e}")
        sys.exit(1)

    # 5. Startup banner
    print("🌐 Backend will be available at:")
    print("   • Local:    http://localhost:8000")
    print("   • Network:  http://0.0.0.0:8000")
    print("\n📖 API Documentation:")
    print("   • Swagger UI: http://localhost:8000/docs")
    print("   • ReDoc:      http://localhost:8000/redoc")
    print("\n🎯 Frontend:")
    print("   • Open frontend.html in your browser")
    print("   • Or visit http://localhost:8000 for embedded frontend")
    print("\n" + "=" * 50)
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    # 6. Launch server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="warning"
    )

if __name__ == "__main__":
    main()
