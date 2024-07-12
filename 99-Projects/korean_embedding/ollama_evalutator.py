# %%
from dotenv import load_dotenv

load_dotenv()

# %%
import autorag
from llama_index.embeddings.upstage import UpstageEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure the Embedding model
upstage_embed_model = UpstageEmbedding()
upstage_embed_model.model_name = "ada"
autorag.embedding_models["solar-embedding-1-large"] = upstage_embed_model
autorag.embedding_models["bge-m3"] = HuggingFaceEmbedding("BAAI/bge-m3")

# %%
from llama_index.llms.upstage import Upstage
from llama_index.llms.ollama import Ollama

# Add LLM models
autorag.generator_models["upstage"] = Upstage
autorag.generator_models["ollama"] = Ollama

# %%
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path="./data/qa.parquet",
    corpus_data_path="./data/corpus.parquet",
    project_dir="./benchmark_ollama",
)

# %%
import nest_asyncio

nest_asyncio.apply()

# %%
evaluator.start_trial("./config/ollama_evaluator_config.yaml")

# %%
