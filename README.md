![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Retrieval Augmented Generation (RAG) Challenges

## Introduction
Retrieval Augmented Generation (RAG) is a novel approach that combines the strengths of retrieval-based and generation-based models to provide accurate and contextually relevant responses. By leveraging a vector database to retrieve relevant documents and a large language model (LLM) to generate responses, RAG can significantly enhance the capabilities of applications in various domains such as customer support, knowledge management, and content creation.

## Project Overview

This project is structured to provide hands-on experience in implementing a RAG system. Students will work through stages from dataset selection to connection to external artefacts (VectorDB, APIs), gaining a comprehensive understanding of RAG’s components and their integration.

### 1. Dataset Selection

Select a dataset suitable for your RAG application. Possible options include:
- **Learning Material**: A collection of books, slide decks on a specific topic
- **News articles**: A dataset containing articles on various topics.
- **Product Reviews**: Reviews of products along with follow-up responses.

### 2. Exploratory Data Analysis (EDA)

Perform an EDA on the chosen dataset to understand its structure, content, and the challenges it presents. Document your findings and initial thoughts on how the data can be leveraged in a RAG system.

### 3.A Embed your chunks of documents:

- **Objective**: Transform your chunks of documents into embeddings that can go into a VectorDB
- **Suggested Tool**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (for english content)

### 3.B Connection to Vector DB

- **Objective**: Connect to a vector database to store and retrieve document embeddings.
- **Suggested Tool**: [ChromaDB](https://www.trychroma.com/)
- **Steps**:
  1. Pre-process the dataset to generate embeddings for each document using a suitable model (e.g., Sentence Transformers).
  2. Store these embeddings in ChromaDB.
  3. Implement retrieval logic to fetch relevant documents based on a query.

### 3.C AI Frameworks

 - Consider using known framework like [LangChain](https://python.langchain.com/docs/integrations/vectorstores/chroma) and [LlamaIndex](https://gpt-index.readthedocs.io/en/latest/examples/vector_stores/ChromaIndexDemo.html) for an easier integration

### 4. Connection to LLM

- **Objective**: Connect to a Large Language Model to generate responses based on retrieved documents.
- **Suggested Tool**: [OpenAI API](https://platform.openai.com/docs/api-reference/introduction)
- **Steps**:
  1. Set up access to the OpenAI API or an alternative LLM API.
  2. Develop the logic to combine retrieved documents with the query to generate a response.
  3. Implement and test the end-to-end RAG pipeline.

### 5. Evaluation

- **Objective**: Evaluate the performance of your RAG system.
- **Metrics**: RAG Systems are not easy to evaluate. Test it yourself a couple of times. BONUS: ask an LLM to create questions and evaluate your RAG's answers
- **Steps**:
  1. Create a test set of queries and expected responses.
  2. Measure the performance of your RAG system against these queries.
  3. Analyze and document the strengths and weaknesses of your system.

### 6. Deployment (Optional)

- **Objective**: Deploy the RAG system as a web application or API.
- **Tools**: Consider using frameworks like Flask or FastAPI for the backend and Streamlit for the frontend.
- **Steps**:
  1. Develop a simple web interface to interact with your RAG system.
  2. Deploy the application on a cloud platform such as AWS, GCP, or Heroku.

## Resources

- [ChromaDB Documentation](https://www.trychroma.com/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/introduction)
- [Sentence Transformers](https://www.sbert.net/)
- [Flask](https://flask.palletsprojects.com/)
- [Streamlit](https://streamlit.io/)

## Deliverables

1. **Python Code**: Provide well-documented Python code implementing the RAG system.
2. **Report**: Submit a detailed report documenting your EDA findings, connection setups, evaluation metrics, and conclusions about the system's performance.
3. **Presentation**: Prepare a short presentation covering the project, from dataset analysis to the final evaluation. Include visual aids such as charts and example responses.

## Bonus

- **Interactive Demo**: Provide an interactive demo of your RAG system during the presentation.

This project will equip you with practical skills in implementing and evaluating a Retrieval Augmented Generation system, preparing you for advanced applications in the field of natural language processing.