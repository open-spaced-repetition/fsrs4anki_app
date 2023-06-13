from zipfile import ZipFile
import os
from pathlib import Path


# Extract the collection file or deck file to get the .anki21 database.
def extract(file, prefix):
    proj_dir = Path(f'projects/{prefix}_{file.orig_name.replace(".", "_").replace("@", "_")}')
    with ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(proj_dir)
        # print(f"Extracted {file.orig_name} successfully!")
    return proj_dir

def cleanup(proj_dir: Path, files):
    """
    Delete all files in prefix that dont have filenames in files
    :param proj_dir:
    :param files:
    :return:
    """
    for file in proj_dir.glob('*'):
        if file.name not in files:
            os.remove(file)

