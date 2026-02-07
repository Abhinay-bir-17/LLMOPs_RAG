from pathlib import Path
from multi_doc_chat.utils.file_io import save_uploaded_files

# simulate an uploaded file object
class DummyFile:
    def __init__(self, name, content):
        self.filename = name
        self.file = content

import io

dummy = DummyFile(
    "test.txt",
    io.BytesIO(b"Hello world")
)   

out_dir = Path("tmp_test_files")
paths = save_uploaded_files([dummy], out_dir)

print("Saved paths:", paths)
assert paths[0].exists()
    