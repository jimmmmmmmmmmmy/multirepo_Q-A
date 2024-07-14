# Architectural Code Q&A Chatbot

This Python application implements a chatbot that answers questions about Github repos

## Features

- Scrapes code from specified GitHub repositories
- Generates embeddings for code snippets using Anthropic's API
- Stores embeddings in a Pinecone vector database for efficient retrieval
- Provides a Streamlit-based user interface for asking questions about the code
- Uses Claude to generate responses based on relevant code snippets

## Dependencies

This code uses the following main libraries:
- `streamlit`: for building the user interface
- `pinecone`: for storing and retrieving relevant code snippets
- `anthropic`: for generating embeddings and responses
- `github`: for scraping code from GitHub repositories
- `pandas`: for data manipulation
- `tiktoken`: for tokenization
- `asyncio`: for asynchronous operations

To install these libraries, use the following command:
```
pip install -r requirements.txt
```

## Setup

1. Clone this repository to your local machine.

2. Create a `.streamlit/secrets.toml` file in the project root directory with the following content:
   ```toml
   [API]
   PINECONE_API_KEY = "your_pinecone_api_key"
   ANTHROPIC_API_KEY = "your_anthropic_api_key"
   GITHUB_TOKEN = "your_github_personal_access_token"
   PINECONE_INDEX_NAME = "your_pinecone_index_name"
   PINECONE_ENV = "your_pinecone_environment"
   ```
   Replace the placeholder values with your actual API keys and configuration.

3. Update the `repositories` list in `scraper-embedder.py` with the GitHub repositories you want to analyze.

4. Run the scraper and embedder:
   ```
   python scraper-embedder.py
   ```
   This will scrape the specified repositories, create embeddings, and store them in Pinecone.

5. Start the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Usage

1. Open a web browser and navigate to `http://localhost:8501` (or the URL provided by Streamlit).
2. Enter your Anthropic API key in the sidebar.
3. Ask questions about the architectural code in the repositories you've scraped.
4. The chatbot will retrieve relevant code snippets and use Claude to generate responses.

## How it Works

1. The scraper-embedder script scrapes code from specified GitHub repositories.
2. It generates embeddings for code snippets using Anthropic's API.
3. The embeddings are stored in a Pinecone vector database.
4. When a user asks a question, the app retrieves relevant code snippets based on similarity search.
5. The chatbot uses Claude to generate a response based on the question and relevant code snippets.
6. The response is displayed to the user in the Streamlit interface.

## Customization

- To add more repositories, update the `repositories` list in `scraper-embedder.py`.
- To modify the code file types being scraped, adjust the file extension check in the `scrape_github_repos` function.
- To change the embedding model or other Anthropic API parameters, modify the relevant sections in `scraper-embedder.py` and `streamlit_app.py`.

## Contributing

Contributions to improve the chatbot or extend its functionality are welcome. Please feel free to submit issues or pull requests.

## License

[Specify your license here]
