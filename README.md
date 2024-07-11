# Coal-Market-Trends-Analyzer


This project implements an advanced query processing system that combines document analysis and news retrieval to provide comprehensive insights for procurement strategies.

## Components

1. `combine_query.py`: Main script that orchestrates the query processing workflow.
2. `news.py`: Handles news article extraction and summarization.
3. `query.py`: Processes queries against the document database.
4. `setup_config.py`: Contains configuration and utility functions for the project.

## Features

- PDF document analysis using language models and vector stores
- Real-time news retrieval and summarization
- Task splitting for complex queries
- Combining insights from document analysis and news articles
- Conversational memory for context-aware responses

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- LangChain
- Chroma
- Tavily API

## Setup

1. Install the required dependencies.
2. Set up your Tavily API key.
3. Prepare your document corpus in the specified directory.

## Data Sources
The PDF document used in this project can be found at the official Argus Media site: [link](https://www.argusmedia.com/-/media/Files/methodology/argus-coal-daily-international.ashx)

## Usage

Run the `combine_query.py` script with your query:

```python
python combine_query.py
```

