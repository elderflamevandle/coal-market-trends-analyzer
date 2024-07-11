import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig, BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import time
from langchain_community.llms import CTransformers
import re

warnings.filterwarnings('ignore')

def initialize_model_and_tokenizer(model_name, cache_dir="/data-disk/models"):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True,
                                                 device_map="auto", quantization_config=quantization_config,
                                                 cache_dir=cache_dir)
    return model, tokenizer

def initialize_pipeline(model, tokenizer, model_name):
    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 1024
    generation_config.temperature = 0.0001
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15

    text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=True,
                                        generation_config=generation_config)
    return text_generation_pipeline

def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/llm-embedder", model_kwargs={"device": "cuda"},
                                 encode_kwargs={"normalize_embeddings": True})

def load_documents_and_prepare_vectorstore(directory_path, glob_pattern="*.pdf", embeddings=None):
    start_time = time.time()  # Start time

    loader = DirectoryLoader(directory_path, glob=glob_pattern, loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts_chunks = text_splitter.split_documents(documents)

    db = Chroma.from_documents(texts_chunks, embeddings, persist_directory="db")

    end_time = time.time()  # End time
    processing_time = end_time - start_time  # Calculate processing time

    print(f"Total processing time: {processing_time:.2f} seconds")

    return db

def split_query_into_tasks(prompt):
    model_path = "/data-disk/aditya_birla/chaitanya_code/model/llama-2-7b-chat.ggmlv3.q8_0.bin"
    llm = CTransformers(model=model_path, model_type='llama', max_new_tokens=512, temperature=0.2)

    prompt_template = f'''SYSTEM: You are a helpful, respectful, and honest assistant. There are two tasks mentioned above, use your intelligence and give me those two tasks on different lines and separate them by |||.
    I want news specific answer in single short sentence format, so that the Tavily model can search for that specific news. Also, do not add any additional information from your side.
    Only give context mentioned in form of two tasks named as Task1 and Task2. Provided Task 2 only specific to news it should not contain anything more. If News is not mentioned in the prompt please assign Task2 as Null

    USER: {prompt}

    ASSISTANT:
    '''

    response = llm(prompt=prompt_template, max_tokens=256, temperature=0.2, top_p=0.95, repeat_penalty=1.2, top_k=150, echo=True)
    
    # Assuming `response` is a dictionary and contains the expected keys
    response_text = response['choices'][0]['text'] if 'choices' in response and len(response['choices']) > 0 else ""
    
    print(response_text)  # For debugging
    
    task1_pattern = r"Task1: (.*?) \|\|\|"
    task2_pattern = r"Task2: (.*?)\."

    task1_context = re.search(task1_pattern, response_text, re.DOTALL)
    task2_context = re.search(task2_pattern, response_text, re.DOTALL)

    task1_text = task1_context.group(1) if task1_context else None
    task2_text = task2_context.group(1) if task2_context else "Null"

    return task1_text, task2_text
