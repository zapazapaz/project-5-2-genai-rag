# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:57:19 2025

@author: igriz
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import openai


# %%
# Define the dataset path
dataset_path = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 8\DAY3\gutenberg_data_cleaned-002.csv"

# Load the dataset
gutenberg = pd.read_csv(dataset_path)

# %%

# Create smaller subset
df = gutenberg.sample(n=500, random_state=42)

# Save smaller dataset
df.to_csv("gutenberg_data_1000.csv", index=False)
# %%
# Display basic information about the dataset
print("Dataset Loaded Successfully!")
print(df.info())
print(df.head())

# Perform Exploratory Data Analysis (EDA)
print("\n--- Dataset Summary ---")
print(df.describe(include='all'))

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Entries ---")
print(df.duplicated().sum())

print("\n--- Column Names ---")
print(df.columns)

# Document findings
eda_report = {
    "num_rows": df.shape[0],
    "num_columns": df.shape[1],
    "missing_values": df.isnull().sum().to_dict(),
    "duplicates": df.duplicated().sum(),
    "column_names": df.columns.tolist()
}

print("\n--- EDA Report ---")
for key, value in eda_report.items():
    print(f"{key}: {value}")

# %%
print(df.columns)
# %%
# Embedding and Storing Chunks
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts, convert_to_tensor=True)


# Ensure 'Text' column is clean
df['Text'] = df['Text'].fillna("").astype(str)

# Apply embeddings only to non-empty text values
df['embeddings'] = df['Text'].apply(lambda x: embed_texts([x])[0].tolist() if x.strip() else None)

# Remove rows with None embeddings before storing in ChromaDB
df = df.dropna(subset=['embeddings'])

print(f"Remaining rows after removing None embeddings: {len(df)}")


# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chromadb_store")
collection = chroma_client.get_or_create_collection(name="gutenberg_embeddings")

for i, row in df.iterrows():
    if row['embeddings']:  # Store only if embeddings are valid
        collection.add(
            ids=[str(i)],
            embeddings=[row['embeddings']],
            metadatas=[{"text": row['Text']}]
        )


print("Embeddings stored successfully in ChromaDB!")
# %% 
# Connect to OpenAI API for response generation
# Initialize OpenAI client
client = openai.OpenAI(api_key="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")

def generate_response(query):
    # Retrieve relevant documents
    results = collection.query(query_embeddings=[embed_texts([query])[0].tolist()], n_results=5)
    retrieved_docs = [item["text"] for item in results["metadatas"][0]]
    
    # **Limit context size** to prevent large requests
    max_chars = 10000  # Set a character limit to avoid excessive tokens
    context = "\n".join(retrieved_docs)[:max_chars]  # Trim long contexts

    # Generate response using OpenAI API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":"You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )

    return response.choices[0].message.content


print("RAG pipeline is ready!")

# %%

# Different contexts provided in the queries

# "You are a helpful assistant."
# "You are a librarian with intimate knowledge of this catalogue. Draw on examples from your catalogue."


test_queries = [
    "What are the main themes in some of Victor Hugos works? Give an example quote for each.",
    #"Summarize the a book written by Zane Grey in 3 sentences or less.",
    #"Briefly summarize some of Honoré de Balzac's popular works.",
    #"Briefly summarize The Spell of Egypt in 3 sentences or less."
]
def evaluate_rag():
    results = {}
    for query in test_queries:
        response = generate_response(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}\n")
        results[query] = response
    return results
evaluate_rag()
# %%
unique = sorted(df['Title'].dropna().unique())  # Remove NaNs and sort alphabetically

for title in unique:
    print(title)

# %%

# %%

