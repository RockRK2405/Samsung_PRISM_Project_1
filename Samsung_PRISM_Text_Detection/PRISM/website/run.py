import sys
import uvicorn
from pathlib import Path

def main():
    """Launch the PRISM FastAPI backend."""
    # Ensure the src directory and project root are in the path
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("Starting PRISM Text Detector Website...")
    print("Backend URL: http://localhost:8000")
    
    # Run uvicorn
    uvicorn.run("website.backend.app:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=[str(project_root / "website")])

if __name__ == "__main__":
    main()
