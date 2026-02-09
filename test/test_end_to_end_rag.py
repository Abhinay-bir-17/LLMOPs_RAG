from pathlib import Path
import io

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.logger import GLOBAL_LOGGER as log

# ---- simulate uploaded file ----
class FakeUpload:
  def __init__(self, path: Path):
      self.filename = path.name
      self.file = open(path, "rb")

def test_end_to_end_rag():
    # 1️⃣ Test document
    test_file = Path("data/test.pdf")
    assert test_file.exists(), "Test document not found"
     
    uploader = FakeUpload(test_file)

    # 2️⃣ INGESTION
    ingestor = ChatIngestor(
        temp_base="test_temp_data",
        faiss_base="test_faiss_index",
        use_session_dirs=True
    )   

    retriever = ingestor.build_retriever(
        uploaded_files=[uploader],
        chunk_size=300,
        chunk_overlap=50,
        k=4
    )   
    
    assert retriever is not None
    log.info("✅ Ingestion completed")

    session_id = ingestor.session_id
    faiss_path = f"test_faiss_index/{session_id}"

    # 3️⃣ RETRIEVAL + CHAT
    rag = ConversationalRAG(session_id=session_id)
    rag.load_retriever_from_faiss(
        index_path=faiss_path,
        k=4,
        search_type="mmr"
    )   

    # 4️⃣ Ask questions
    questions = [
        "What is the use of add and norm?",
        "What is cross attention?",
        "is cross attention also multi head?"
    ]   

    print('going to loop over all questions')
    for q in questions:
        answer = rag.invoke(q)
        print("\nQ:", q)
        print("A:", answer)

    # 5️⃣ Cleanup
    uploader.file.close()


if __name__ == "__main__":
    test_end_to_end_rag()
