from typing import Callable, Optional
import os
import sys
import shutil
import subprocess  

def open_output_folder(path: str, log: Optional[Callable[[str], None]] = None) -> None:
    
    if log is None:
        log = lambda *_args, **_kwargs: None  


    log(f"[System] Container output folder: {path}")

    
    host_path = None
    ws = os.environ.get("LOCAL_WORKSPACE_FOLDER")
    if ws and path.startswith("/workspace"):
        rel = path[len("/workspace"):].lstrip("/")  
        host_path = os.path.join(ws, rel)
        log(f"[System] Host output folder: {host_path}")
    else:
        log("[System] LOCAL_WORKSPACE_FOLDER not set; cannot compute host path automatically.")

    log("[System] To open the folder, copy the host path above into your file explorer or VS Code.")