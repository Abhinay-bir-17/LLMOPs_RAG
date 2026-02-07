from pathlib import Path
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor

# ---- simulate uploaded file object ----
class FakeUpload:
    def __init__(self, path: Path):
        self.filename = path.name
        self.file = open(path, "rb")

# ---- test ingestion ----
def test_data_ingestion():
    test_file = Path("data/agentic_ai.txt")

    uploader = FakeUpload(test_file)

    ingestor = ChatIngestor(
        temp_base="test_temp_data",
        faiss_base="test_faiss_index",
        use_session_dirs=True
    )       

    retriever = ingestor.build_retriever(
        uploaded_files=[uploader],
        chunk_size=50,
        chunk_overlap=20,
        k=3
    )

    assert retriever is not None
    print("✅ Retriever built successfully")
    docs = retriever.get_relevant_documents("What is Agentic AI?")
    print("doc[0].page_content:",docs[0].page_content)
    # cleanup
    uploader.file.close()

if __name__ == "__main__":
    test_data_ingestion()
