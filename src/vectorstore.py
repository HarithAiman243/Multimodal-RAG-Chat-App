import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

class VectorDB:
    """
    A connector class for a pre-populated Pinecone vector database.
    Its sole purpose is to connect to the index and provide a retriever.
    """
    def __init__(self):
        """
        Initializes the connection to the Pinecone index.
        """
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set in the .env file.")

        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME is not set in the .env file.")

        # Initialize Pinecone client
        pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Verify that the index exists
        if self.index_name not in pc.list_indexes().names():
            raise ValueError(f"Pinecone index '{self.index_name}' not found. Please run ingest.py first.")

        # Initialize the embedding model
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

        # Initialize the LangChain PineconeVectorStore wrapper
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embedding_model
        )

    def as_retriever(self, search_kwargs={'k': 5}):
        """
        Returns the vector store instance configured as a retriever.
        
        Args:
            search_kwargs (dict): A dictionary to configure search parameters,
                                  such as the number of documents to retrieve ('k').
        
        Returns:
            A LangChain retriever object.
        """
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)