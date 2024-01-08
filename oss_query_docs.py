"""
In the same directory:

git clone https://github.com/holoviz/lumen.git
mkdir data
cp lumen/doc/**/*.md data
cp lumen/doc/*.md data
          
      
       

"""

from transformers import AutoTokenizer
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.embeddings import HuggingFaceEmbedding

from panel.chat import ChatInterface

BASE_REPO = "HuggingFaceH4/zephyr-7b-beta"
QUANTIZED_REPO = "TheBloke/zephyr-7B-beta-GGUF"
QUANTIZED_FILE = "zephyr-7b-beta.Q5_K_S.gguf"
DATA_DIR = "data"


def load_llm():
    model_url = f"https://huggingface.co/{QUANTIZED_REPO}/resolve/main/{QUANTIZED_FILE}"
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_REPO)


def create_query_engine(llm):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    documents = SimpleDirectoryReader(DATA_DIR, required_exts=[".md"]).load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine(streaming=True)
    return query_engine


async def respond(contents, user, instance):
    conversation = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who will help the user.",
        },
        {"role": "user", "content": contents},
    ]
    formatted_contents = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    response = query_engine.query(formatted_contents)
    message = None
    for chunk in response.response_gen:
        message = instance.stream(chunk, message=message, user="Assistant")


llm = load_llm()
tokenizer = load_tokenizer()
query_engine = create_query_engine(llm)
chat_interface = ChatInterface(callback=respond, callback_exception="verbose")
chat_interface.send(
    "This uses Llama Index to search through Lumen docs. Try asking a question, "
    "like: 'How do I add a hvplot view to penguins.csv?'",
    user="System",
    respond=False,
)
chat_interface.servable()