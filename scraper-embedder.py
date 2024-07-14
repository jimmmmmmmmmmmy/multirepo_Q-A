import os
import time
from uuid import uuid4

import pandas as pd
import pinecone
import tiktoken
import toml
from github import Github
from langchain.text_splitter import RecursiveCharacterTextSplitter
from anthropic import Anthropic
from tqdm.auto import tqdm

# Load the secrets.toml file
secrets = toml.load(".streamlit/secrets.toml")

PINECONE_API_KEY = secrets["API"]["PINECONE_API_KEY"]
ANTHROPIC_API_KEY = secrets["API"]["ANTHROPIC_API_KEY"]
GITHUB_TOKEN = secrets["API"]["GITHUB_TOKEN"]

client = Anthropic(api_key=ANTHROPIC_API_KEY)

def read_repositories_from_md():
    with open('case_studies.md', 'r') as file:
        content = file.read()
    
    # Use regex to find all GitHub repository URLs
    repo_urls = re.findall(r'https://github.com/([^/]+/[^/\)]+)', content)
    return repo_urls

repositories = read_repositories_from_md()

def scrape_github_repos(repositories):
    """Scrape code from GitHub repositories"""
    g = Github(GITHUB_TOKEN)
    code_data = []

    for repo_name in repositories:
        repo = g.get_repo(repo_name)
        contents = repo.get_contents("")

        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            else:
                if file_content.name.endswith(('.py', '.js', '.java', '.cpp', '.cs', '.go')):  # Add more extensions as needed
                    code_data.append({
                        "repo": repo_name,
                        "file_path": file_content.path,
                        "content": file_content.decoded_content.decode('utf-8')
                    })

    return pd.DataFrame(code_data)

# Create the length function
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Use the appropriate tokenizer for Claude
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

def create_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for idx, record in enumerate(tqdm(data)):
        texts = text_splitter.split_text(record["content"])
        chunks.extend([
            {
                "id": str(uuid4()),
                "repo": record["repo"],
                "file_path": record["file_path"],
                "text": texts[i],
                "chunk": i
            }
            for i in range(len(texts))
        ])
    return chunks

def init_pinecone(api_key, environment):
    pinecone.init(api_key=api_key, environment=environment)

def create_index_if_not_exists(index_name, dimension, metric):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension, metric=metric)
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

def create_embeddings(chunks, index, batch_size=100):
    for i in tqdm(range(0, len(chunks), batch_size)):
        i_end = min(len(chunks), i + batch_size)
        meta_batch = chunks[i:i_end]
        ids_batch = [x["id"] for x in meta_batch]
        texts = [x["text"] for x in meta_batch]
        try:
            res = anthropic.embeddings.create(
                model="claude-3-sonnet-20240229",
                input=texts
            )
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            time.sleep(5)
            continue

        embeds = [record.embedding for record in res.data]
        meta_batch = [{
            "repo": x["repo"],
            "file_path": x["file_path"],
            "text": x["text"],
            "chunk": x["chunk"]
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        index.upsert(vectors=to_upsert)

def main():
    print("Scraping GitHub repositories...")
    data = scrape_github_repos(repositories)

    print("Creating chunks...")
    chunks = create_chunks(data.to_dict('records'))

    print("Initializing Pinecone...")
    PINECONE_INDEX_NAME = secrets["API"]["PINECONE_INDEX_NAME"]
    PINECONE_ENV = secrets["API"]["PINECONE_ENV"]

    init_pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    create_index_if_not_exists(
        index_name=PINECONE_INDEX_NAME,
        dimension=1536,  # Dimension for Claude's text embedding model
        metric="cosine"
    )

    index = pinecone.Index(PINECONE_INDEX_NAME)

    print("Creating embeddings and uploading to Pinecone...")
    create_embeddings(
        chunks=chunks,
        index=index,
        batch_size=100
    )

    print("Process completed successfully!")

if __name__ == "__main__":
    main()
