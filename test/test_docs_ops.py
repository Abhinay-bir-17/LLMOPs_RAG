from multi_doc_chat.utils.document_ops import load_documents
from pathlib import Path

test_file = Path("tmp_test_files/test.txt")

docs = load_documents([test_file])
print(docs)

assert len(docs) == 1
assert docs[0].page_content.strip() == "Hello world"
