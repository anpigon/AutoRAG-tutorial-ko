# %%
import nest_asyncio
from dotenv import load_dotenv

load_dotenv()

import autorag
from autorag import web

from llama_index.embeddings.upstage import UpstageEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

nest_asyncio.apply()

upstage_embed_model = UpstageEmbedding()
upstage_embed_model.model_name = "ada"
autorag.embedding_models["solar-embedding-1-large"] = upstage_embed_model
autorag.embedding_models["bge-m3"] = HuggingFaceEmbedding("BAAI/bge-m3")

# %%
runner = web.get_runner(yaml_path="", project_dir="", trial_path="./benchmark/0")
web.set_initial_state()
web.set_page_config()
web.set_page_header()
web.chat_box(runner)

# %%
