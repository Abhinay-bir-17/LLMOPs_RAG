from multi_doc_chat.utils.model_loader import ModelLoader
         
ml = ModelLoader()
         
emb = ml.load_embeddings()
print("Embedding dim:", len(emb.embed_query("hello")))
         
llm = ml.load_llm()
print("LLM response:", llm.invoke("Say hello").content)
         