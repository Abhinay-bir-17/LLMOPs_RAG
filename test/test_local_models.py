from langchain_ollama import OllamaEmbeddings, ChatOllama

emb = OllamaEmbeddings(model="nomic-embed-text")
print(len(emb.embed_query("hello world")))

llm = ChatOllama(model="qwen2.5:0.5b")
print(llm.invoke("explain mixture of experts in LLaMA").content)
