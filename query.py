from setup_config import *
import textwrap
from langchain.chains import LLMChain

# Initialize everything once
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)
pipeline = initialize_pipeline(model, tokenizer, MODEL_NAME)
embeddings = initialize_embeddings()
db = load_documents_and_prepare_vectorstore("/data-disk/aditya_birla/my_doc", embeddings=embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

pdf_prompt = """Act as procurement agent,you will always look out for the prices and draw inferences.If you cannnot find the number, proof read it. Given the
following conversation and a follow up question, rephrase the follow up question
to be a standalone question. At the end of standalone question add this
'Answer the question in English language.' If you do not know the answer reply with 'I am sorry, I dont have enough information'.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(pdf_prompt)

def process_query(query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=HuggingFacePipeline(pipeline=pipeline),
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        )

    result_ = qa_chain({"question": query})
    result = result_["answer"].strip()
    wrapped_result = textwrap.fill(result, width=100)
    print(wrapped_result)

if __name__ == "__main__":
    #query = "Please extract fob newcastle price for australia from january 2023"
    query1 = "this is the latest article..Australian coal with an energy content of 5,500 kcal/kg was priced at $93.28 a metric ton in the week to Dec. 8, indicating a slight decrease from its recent high of $105. Tell me effect it might have on the future prices"
    process_query(query1)


