# src/vectorstore.py

import os
import shutil
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager

from src.utils import load_config

load_dotenv()

class VectorDB:
    """
    A class to connect to a persistent, pre-populated Pinecone vector database.
    
    This class is a 'read-only' client. It assumes that an external data pipeline
    is responsible for creating the index and populating it with data from the
    Ads Manager API.
    
    It does NOT contain logic to index new documents or delete the index.
    """
    def __init__(self):
        config = load_config()
        self.cache_dir = './.cache/database'
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Initialize the Embedding Model
        embedding = OpenAIEmbeddings(model=config['embedding_model']['model_name'])
        
        # 2. Connect to Pinecone
        index_name = config['pinecone_index_name']
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        
        # --- CRITICAL CHANGE ---
        # We no longer delete or create the index here. We only check if it exists.
        # If it doesn't exist, it means the data pipeline hasn't been run, which is an error.
        if index_name not in pc.list_indexes().names():
            raise ValueError(
                f"Pinecone index '{index_name}' not found! "
                "Please run your data pipeline script first to create and populate the index."
            )
        
        # 3. Initialize the Pinecone Vector Store object
        self.vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)
        
        # 4. Initialize Record Manager (still useful for some advanced LangChain features, but not for indexing here)
        namespace = f'pinecone/{index_name}'
        self.record_manager = SQLRecordManager(
            namespace, db_url=f'sqlite:///{self.cache_dir}/record_manager_cache.sql'
        )
        self.record_manager.create_schema()

    def as_retriever(self):
        """
        Returns the vector store instance as a retriever for the RAG chain.
        """
        # You can add retriever-specific configurations here if needed
        # For example: search_kwargs={'k': 5}
        return self.vectorstore.as_retriever()

    def __del__(self):
        """
        Cleans up the local cache directory when the object is destroyed.
        """
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)