# src/openai_chain.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.vectorstore import VectorDB
from src.utils import load_config

class OpenAIRAGChain:
    """
    A class that encapsulates the logic for a conversational RAG (Retrieval-Augmented Generation) chain.
    
    This chain connects to a pre-populated VectorDB, retrieves relevant context based on user queries,
    and uses an LLM to generate answers grounded in that context. It is designed to be stateful,
    maintaining a history of the conversation.
    """
    def __init__(self, chat_memory):
        config = load_config()
        self.chat_memory = chat_memory

        # 1. Initialize VectorDB and Retriever
        # This connects to the existing Pinecone index.
        self.vector_db = VectorDB()
        retriever = self.vector_db.as_retriever()

        # 2. Initialize the LLM
        llm = ChatOpenAI(
            model=config['chat_model']['model_name'],
            temperature=config['chat_model']['temperature']
        )

        # 3. Define the System Prompts
        
        # This prompt helps the LLM rephrase a follow-up question to be a standalone question
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # This is the main prompt for answering the question based on retrieved context
        qa_system_prompt = (
            "You are an expert AI assistant for answering questions about Facebook Ads Manager data. "
            "You MUST base your answer EXCLUSIVELY on the following pieces of retrieved context. "
            "Format your answers clearly, using bullet points or tables where it makes sense. "
            "If the context does not contain the answer, say 'The provided data does not contain enough "
            "information to answer this question.' Do NOT use any outside knowledge.\n\n"
            "Context:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 4. Construct the RAG Chain
        
        # Chain to create a standalone question
        history_aware_retriever = contextualize_q_prompt | llm | StrOutputParser() | retriever

        # Chain to answer the question
        question_answer_chain = (
            RunnablePassthrough.assign(context=history_aware_retriever)
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        # 5. Wrap the chain with message history management
        self.chain_with_history = RunnableWithMessageHistory(
            question_answer_chain,
            lambda session_id: self.chat_memory, # Use the passed-in chat memory object
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def run(self, user_input: str):
        """
        Invokes the RAG chain with the user's input.
        
        The chat history is managed automatically by the RunnableWithMessageHistory wrapper.
        A dummy session_id is used because StreamlitChatMessageHistory manages history
        based on its unique key, not the session_id.
        """
        response = self.chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default_session"}}
        )
        return response