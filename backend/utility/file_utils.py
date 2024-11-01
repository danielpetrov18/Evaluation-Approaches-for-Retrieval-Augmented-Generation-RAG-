from pathlib import Path
from typing import Iterator

def iterate_over_files(folder_path: str | Path) -> Iterator[str]:
    """
    Iterate over all files in a given folder and its sub folders.

    Args:
        folder_path: Path to the folder to iterate over.

    Yields:
        Path to each file.

    Returns:
        Iterator of file paths.
    """
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    
    return (str(file_path) for file_path in folder_path.rglob('*') if file_path.is_file())