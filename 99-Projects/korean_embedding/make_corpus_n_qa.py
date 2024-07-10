# %%
import os
import pandas as pd
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter

from autorag.utils import cast_corpus_dataset
from autorag.data.corpus import llama_text_node_to_parquet
from autorag.data.qacreation import generate_qa_llama_index, make_single_content_qa

load_dotenv()

root_dir = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(root_dir, "raw_docs")
corpus_path = os.path.join(root_dir, "data", "corpus.parquet")
qa_path = os.path.join(root_dir, "data", "qa.parquet")

# %%
# Create a corpus
documents = SimpleDirectoryReader(dir_path, recursive=True).load_data()
nodes = TokenTextSplitter().get_nodes_from_documents(
    documents=documents, chunk_size=256, chunk_overlap=64
)
corpus_df = llama_text_node_to_parquet(nodes)
corpus_df = cast_corpus_dataset(corpus_df)

if os.path.isfile(corpus_path):
    os.remove(corpus_path)

corpus_df.to_parquet(corpus_path)


# %%
# Create QA pairs
prompt = """
You are tasked with generating question and answer pairs based on a given passage. The questions should be written in Korean, while the answers should be in English.

Here is the passage to analyze:
<passage>
{{text}}
</passage>

You are to generate {{num_questions}} question and answer pairs based on this passage.

When generating the questions and answers, follow these guidelines:
1. Ensure that the questions are relevant to the main ideas and details in the passage.
2. Write the questions in Korean.
3. Provide the answers in Korean.
4. Make sure the answers are accurate and directly supported by the information in the passage.
5. Vary the types of questions (e.g., factual, inferential, vocabulary-based) to cover different aspects of the text.

Use the following format for each question-answer pair:

[Q]: (Korean question here)
[A]: (Korean answer here)

Here's an example of how your output should look:

[Q]: 이 글의 주제는 무엇인가요?
[A]: 이 구절의 주요 주제는 규칙적인 운동의 중요성입니다.

Please generate the specified number of question-answer pairs, numbering them sequentially. Begin your response with "Generated Questions and Answers:" and then list the pairs.
"""

corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")

# llm = OpenAI(model="gpt-4o", temperature=0.5)
llm = Ollama(model="gemma2", temperature=1.0)

if os.path.isfile(qa_path):
    os.remove(qa_path)

qa_df = make_single_content_qa(
    corpus_df,
    content_size=49,
    qa_creation_func=generate_qa_llama_index,
    llm=llm,
    prompt=prompt,
    question_num_per_content=1,
    output_filepath=qa_path,
)
