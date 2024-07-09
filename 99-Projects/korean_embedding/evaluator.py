# %%
from dotenv import load_dotenv

load_dotenv()

import autorag
from autorag.evaluator import Evaluator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.upstage import UpstageEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.upstage import Upstage


# Configure the Embedding model
upstage_embed_model = UpstageEmbedding()
upstage_embed_model.model_name = "ada"
autorag.embedding_models["solar-embedding-1-large"] = upstage_embed_model
autorag.embedding_models["bge-m3"] = HuggingFaceEmbedding("BAAI/bge-m3")

# Add LLM models
autorag.generator_models["upstage"] = Upstage

evaluator = Evaluator(
    qa_data_path="./data/qa.parquet",
    corpus_data_path="./data/corpus.parquet",
    project_dir="./benchmark",
)

evaluator.start_trial("./config/korean_embedding.yaml")

# %%

