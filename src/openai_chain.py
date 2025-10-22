import yaml
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from src.vectorstore import VectorDB

class RAGChain:
    """
    Encapsulates the complete RAG logic, from retrieval to generation.
    """
    def __init__(self, chat_memory_history):
        """
        Initializes the RAG chain with all necessary components.
        
        Args:
            chat_memory_history: A LangChain chat message history object.
        """
        # 1. Load configuration
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # 2. Initialize components
        vector_db = VectorDB()
        self.retriever = vector_db.as_retriever()
        self.chat_memory = chat_memory_history
        
        llm = ChatOpenAI(
            model=config['llm']['model_name'],
            temperature=config['llm']['temperature']
        )

        # 3. Define the prompt template
        # This prompt uses the system message from config and structures the inputs.
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", config['llm']['system_prompt']),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # 4. Construct the main RAG chain
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: self.format_docs(x["context"]))
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        
        self.chain_with_history = RunnableWithMessageHistory(
            RunnablePassthrough.assign(
                context=self.contextualized_question | self.retriever
            ) | rag_chain_from_docs,
            lambda session_id: self.chat_memory,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    @staticmethod
    def format_docs(docs):
        """Helper function to format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    @property
    def contextualized_question(self):
        """
        Creates a sub-chain to rephrase a follow-up question into a standalone question
        using the conversation history.
        """
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        contextualize_q_llm = ChatOpenAI(temperature=0)
        return contextualize_q_prompt | contextualize_q_llm | StrOutputParser()

    def run(self, user_input: str):
        """
        Invokes the RAG chain with the user's input and manages history.
        """
        # A dummy session_id is used because StreamlitChatMessageHistory is managed by its key.
        return self.chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default_session"}}
        )