from transformers import AutoTokenizer, pipeline
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langchain_community.llms import CTransformers
from transformers import BartForConditionalGeneration, BartTokenizer
from tavily import TavilyClient

class NewsArticleExtractor:
    def __init__(self, client):
        self.client = client

    def search_articles(self, prompt):
        # Use the search tool to invoke a search.
        search_results = self.client.search(prompt, 
                         search_depth="advanced",
                         topic="news",
                         include_answer=True,
                         )
        print(search_results)

        return search_results

class TextSummarizationPipeline():
    def __init__(self, model_name, max_length):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def summarize(self, text, maxSummarylength=500):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=maxSummarylength, min_length=int(maxSummarylength/5), length_penalty=10.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def split_text_into_pieces(self, text, max_tokens=900, overlapPercent=10):
        tokens = self.tokenizer.tokenize(text)
        overlap_tokens = int(max_tokens * overlapPercent / 100)
        pieces = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap_tokens)]
        text_pieces = [self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(piece), skip_special_tokens=True) for piece in pieces]
        return text_pieces

    def recursive_summarize(self, text, max_length, recursionLevel=0):
        recursionLevel += 1
        tokens = self.tokenizer.tokenize(text)
        expectedCountOfChunks = len(tokens) / max_length
        max_length = int(len(tokens) / expectedCountOfChunks) + 2

        pieces = self.split_text_into_pieces(text, max_tokens=max_length)
        print("Number of pieces: ", len(pieces))
        
        summaries = []
        for k, piece in enumerate(pieces):
            print("****************************************************")
            print(f"Piece: {k+1} out of {len(pieces)} pieces\n{piece}\n")
            summary = self.summarize(piece, maxSummarylength=max_length/3*2)
            print("SUMMARY: ", summary)
            summaries.append(summary)
            print("****************************************************")

        concatenated_summary = ' '.join(summaries)
        tokens = self.tokenizer.tokenize(concatenated_summary)

        if len(tokens) > max_length:
            print("############# GOING RECURSIVE ##############")
            return self.recursive_summarize(concatenated_summary, max_length=max_length, recursionLevel=recursionLevel)
        else:
            final_summary = concatenated_summary
            if len(pieces) > 1:
                final_summary = self.summarize(concatenated_summary, maxSummarylength=max_length)
            return final_summary

class Supervisor:
    def __init__(self, summary_model_name, api_key):
        # import and connect
        self.client = TavilyClient(api_key)

        # Initialize the article extractor and summarizer.
        self.extractor = NewsArticleExtractor(self.client)
        print("\n\n tavity loaded \n\n")
        input("Stopped at Tavity loaded press enter to continue")
        
        self.text_summarizer = TextSummarizationPipeline(summary_model_name, max_length=300)
        print("Text Summarizer Loaded")

    def extract_and_summarize_news(self, query):
        # Extract articles.
        response_news = self.extractor.search_articles(query)
        
        final_news = ""
        # Extract and print the content of each article
        for article in response_news["results"]:
            print(article)
            print(article["content"])
            print("\n---\n")  # Separates each article content for clarity
            final_news += article["content"] + " "
        
        summary = self.text_summarizer.recursive_summarize(final_news, max_length=500)
        return summary

if __name__ == "__main__":
    # Example usage of the Supervisor class.
    summary_model = 'facebook/bart-large-cnn'
    tavity_api_key = "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"  # Replace with your actual Tavily API key
    supervisor = Supervisor(summary_model_name=summary_model, api_key="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")

    # Example query for news articles.
    query = "what is?"

    # Run the news extraction and summarization.
    final_summary = supervisor.extract_and_summarize_news(query)
    print(final_summary)
