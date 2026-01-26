import os
import sys
import json
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.exception import DocumentPortalException

class ApiKeyManager:
    REQUIRED_KEYS = ["GOOGLE_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("apikeyliveclass")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))




        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # Final check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)

        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})


    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))



    def load_embeddings(self):
      """
      Load and return embedding model based on environment:
      - LOCAL      → Ollama (nomic-embed-text)
      - PRODUCTION → Google Gemini
      """
      try:
          env = os.getenv("ENV", "local").lower()

          if env != "production":
              # -------- LOCAL: Ollama embeddings --------
              from langchain_community.embeddings import OllamaEmbeddings

              ollama_cfg = self.config["embedding_model"]["ollama"]
              model_name = ollama_cfg["model_name"]
              base_url = ollama_cfg.get("base_url", "http://localhost:11434")

              log.info(
                  "Loading LOCAL embedding model (Ollama)",
                  model=model_name,
                  base_url=base_url,
              )

              return OllamaEmbeddings(
                  model=model_name,
                  base_url=base_url,
              )

          else:
              # -------- PRODUCTION: Gemini embeddings --------
              from langchain_google_genai import GoogleGenerativeAIEmbeddings

              google_cfg = self.config["embedding_model"]["google"]
              model_name = google_cfg["model_name"]

              log.info(
                  "Loading PRODUCTION embedding model (Gemini)",
                  model=model_name,
              )

              return GoogleGenerativeAIEmbeddings(
                  model=model_name,
                  google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
              )

      except Exception as e:
          log.error("Error loading embedding model", error=str(e))
          raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
      """
      Load and return LLM based on environment:
      - LOCAL      → Ollama (qwen2.5:0.5b)
      - PRODUCTION → Google Gemini 2.5 Flash
      """
      try:
          env = os.getenv("ENV", "local").lower()
          llm_block = self.config["llm"]

          if env != "production":
              # -------- LOCAL: Ollama (qwen2.5:0.5b) --------
              from langchain_community.chat_models import ChatOllama

              ollama_cfg = llm_block["ollama"]
              model_name = ollama_cfg["model_name"]
              temperature = ollama_cfg.get("temperature", 0)
              max_tokens = ollama_cfg.get("max_output_tokens", 2048)
              base_url = ollama_cfg.get("base_url", "http://localhost:11434")

              log.info(
                  "Loading LOCAL LLM (Ollama)",
                  model=model_name,
                  base_url=base_url,
              )

              return ChatOllama(
                  model=model_name,
                  temperature=temperature,
                  num_predict=max_tokens,
                  base_url=base_url,
              )

          else:
              # -------- PRODUCTION: Gemini 2.5 Flash --------
              from langchain_google_genai import ChatGoogleGenerativeAI

              google_cfg = llm_block["google"]
              model_name = google_cfg["model_name"]
              temperature = google_cfg.get("temperature", 0)
              max_tokens = google_cfg.get("max_output_tokens", 2048)

              log.info(
                  "Loading PRODUCTION LLM (Gemini)",
                  model=model_name,
              )

              return ChatGoogleGenerativeAI(
                  model=model_name,
                  google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                  temperature=temperature,
                  max_output_tokens=max_tokens,
              )

      except Exception as e:
          log.error("Error loading LLM", error=str(e))
          raise DocumentPortalException("Failed to load LLM", sys)

if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")