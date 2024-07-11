from setup_config import *
import textwrap
from langchain.chains import LLMChain
# Ensure news.py is in the same directory or in the PYTHONPATH
from news import Supervisor as NewsSupervisor
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
import re

# Initialize everything once for PDF processing
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)
pipeline = initialize_pipeline(model, tokenizer, MODEL_NAME)
embeddings = initialize_embeddings()
db = load_documents_and_prepare_vectorstore("/data-disk/aditya_birla/my_doc/pdf_docs", embeddings=embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

pdf_prompt = """
Act as procurement agent, you will always look out for the prices and draw inferences. If you cannot find the number, proofread it. Given the
following conversation and a follow-up question, rephrase the follow-up question
to be a standalone question. At the end of the standalone question add this
'Answer the question in English language.' If you do not know the answer reply with 'I am sorry, I don't have enough information'.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(pdf_prompt)

def process_pdf_query(query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=HuggingFacePipeline(pipeline=pipeline),
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
    )

    result_ = qa_chain({"question": query})
    pdf_inference = result_["answer"].strip()
    return pdf_inference

# Initialize NewsSupervisor with your actual Tavily API key
news_supervisor = NewsSupervisor(summary_model_name='facebook/bart-large-cnn', api_key="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")




def combine_inferences(news_inference, pdf_inference):
    combined_inference_query = f"Considering insights from recent news: {news_inference} and our document analysis: {pdf_inference}, what are the implications for our procurement strategy?"
    # Process the combined inference query for final insights
    final_inference = process_pdf_query(combined_inference_query)  # Reusing the PDF query processing for demonstration
    return final_inference
    
    

def process_combined_query2(original_query):
    task_1_text, task_2_text = split_query_into_tasks(original_query)
    
    # Process each task
    pdf_inference = process_pdf_query(task_1_text) if task_1_text else "No PDF-related task found."
    news_inference = news_supervisor.extract_and_summarize_news(task_2_text) if task_2_text else "No news-related task found."
    

    print(pdf_inference)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(news_inference)
    # Combine inferences
    combined_inference = f"News Inference: {news_inference}\nPDF Inference: {pdf_inference}"
    
    # Optionally create a more synthesized conclusion if both inferences are available
    if task_1_text and task_2_text:
        final_inference = combine_inferences(news_inference, pdf_inference)
        print(textwrap.fill(final_inference, width=100))
    else:
        print(textwrap.fill(combined_inference, width=100))
           
def query_splitter(prompt):
    model_path = r"/data-disk/aditya_birla/chaitanya_code/model/llama-2-7b-chat.ggmlv3.q8_0.bin"
    llm = CTransformers(model=model_path, 
                        model_type='llama', 
                        max_new_tokens=512, 
                        temperature=0.1)
    
# Improved prompt template
    prompt_template = f'''SYSTEM: You are a helpful, respectful, and honest assistant.
    Your task is to analyze the USER's prompt and identify specific tasks mentioned within. 
    Please format your response by labeling each task clearly as "Task 1", "Task 2", etc. 
    This will help in separating and identifying the tasks more effectively. 
    Remember to focus only on the tasks directly mentioned by the USER, without adding any external information or analysis.
    
    USER: {prompt}
    
    ASSISTANT:'''
    
    response = llm(prompt=prompt_template, max_tokens=256, top_p=0.95, repeat_penalty=1.2, top_k=150, echo=True)
    
    # Process the response to extract tasks
    tasks = [line.strip() for line in response.split("\n") if "Task 1:" in line or "Task 2:" in line]
    
    task_1, task_2 = "", ""
    for task in tasks:
        if "Task 1:" in task:
            task_1 = task.split("Task 1:", 1)[1].strip()
        elif "Task 2:" in task:
            task_2 = task.split("Task 2:", 1)[1].strip()
    
    print("Response -->", response)
    print("\n")
    print("Tasks -->", tasks)
    print("\n")
    
    task_1, task_2 = "", ""
    for task in tasks:
        if task.lower().startswith("task 1:"):
            task_1 = task.split(":", 1)[1].strip()
        elif task.lower().startswith("task 2"):
            task_2 = task.split(":", 1)[1].strip()
    
    print("Task 1 -->", task_1)
    print("Task 2 -->", task_2)
    
    return task_1, task_2

def process_combined_query(original_query):
    # Split the original query into parts for news and PDF processing
    # This is a placeholder; you'll need to implement actual logic based on your application's needs
    
    task_1_text, task_2_text = query_splitter(original_query)
    print(task_1_text, task_2_text)
    

    pdf_inference = process_pdf_query(task_1_text) if task_1_text else "No PDF-related task found."
    news_inference = news_supervisor.extract_and_summarize_news(task_2_text) if task_2_text else "No news-related task found."
    
    # Combine inferences
    combined_inference = f"News Inference: {news_inference}\nPDF Inference: {pdf_inference}"
    
    # Optionally create a more synthesized conclusion if both inferences are available
    if task_1_text and task_2_text:
        final_inference = combine_inferences(news_inference, pdf_inference)
        print(textwrap.fill(final_inference, width=100))
    else:
        print(textwrap.fill(combined_inference, width=100))
        
        
def process_combine_query1(original_query):
    # Split the original query into parts for news and PDF processing
    # This is a placeholder; you'll need to implement actual logic based on your application's needs
    task_1_text, task_2_text = original_query.split(" | ")
    # pdf_query , news_query, = original_query.split(" | ")
    # Process news-related part

    pdf_inference = process_pdf_query(task_1_text) if task_1_text else "No PDF-related task found."
    news_inference = news_supervisor.extract_and_summarize_news(task_2_text) if task_2_text else "No news-related task found."
    
    # Combine inferences
    combined_inference = f"News Inference: {news_inference}\nPDF Inference: {pdf_inference}"
    
    # Optionally create a more synthesized conclusion if both inferences are available
    if task_1_text and task_2_text:
        final_inference = combine_inferences(news_inference, pdf_inference)
        print(textwrap.fill(final_inference, width=100))
    else:
        print(textwrap.fill(combined_inference, width=100))
    
    
    
    
if __name__ == "__main__":
    original_query1 = "Please extract fob newcastle price of coal from January 2023  to may 2023 . See which factors have impacted the trend of the extracted prices | Fetch the latest news article on coal in Australia and derive if there Australian coal price is going up or down"
    
    original_query = "Please extract fob newcastle price of coal from January 2023  to may 2023 . See which factors have impacted the trend of the extracted prices. Fetch the latest news article on coal in Australia and derive if there Australian coal price is going up or down"
    process_combined_query(original_query)
