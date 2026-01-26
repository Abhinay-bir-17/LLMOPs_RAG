import os
from dotenv import load_dotenv

# Load env vars locally (safe for testing)
load_dotenv()

print("ENV =", os.getenv("ENV"))

# --------- Test Gemini Embeddings ---------
from langchain_google_genai import GoogleGenerativeAIEmbeddings

print("\nTesting Gemini embeddings...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

vector = embeddings.embed_query("hello world")
print("Embedding vector length:", len(vector))


# --------- Test Gemini LLM ---------
from langchain_google_genai import ChatGoogleGenerativeAI

print("\nTesting Gemini LLM (2.5 Flash)...")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_output_tokens=200,
)

response = llm.invoke("Explain Agentic AI in one sentence.")
print("LLM response:", response.content)
