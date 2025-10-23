import streamlit as st
import yaml
import uuid
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from src.openai_chain import RAGChain
from src.utils import save_chat_history, load_chat_history, get_saved_sessions

# --- Page Configuration ---
st.set_page_config(page_title="Multimodal RAG LLM Chatbot", layout="wide")

def check_password():
    """Returns `True` if the user enters the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password.
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("😕 Password incorrect.")
    return False

def run_app():
    """
    The main function that runs the chatbot application after successful authentication.
    """
    # --- Session State Initialization ---
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # --- Sidebar for Conversation Management ---
    with st.sidebar:
        st.title("Conversation History")
        
        if st.button("➕ New Chat"):
            st.session_state.session_id = str(uuid.uuid4())
            # Use st.rerun() to immediately reflect the new chat session
            st.rerun()

        st.markdown("---")

        # --- Language Selection ---
        st.markdown("## Language Settings")
        output_language = st.selectbox(
            "Select response language:",
            ("English", "Malay", "Mandarin")
        )
        # --------------------------

        st.markdown("---")
        
        saved_sessions = get_saved_sessions()
        if not saved_sessions:
            st.write("No saved conversations yet.")
        else:
            st.markdown("Past Conversations:")
            for session in saved_sessions:
                session_id = session['session_id']
                # Create a simple, readable title from the session ID and date
                title = f"Chat from {session['last_modified'].strftime('%Y-%m-%d %H:%M')}"
                if st.button(title, key=session_id, use_container_width=True):
                    st.session_state.session_id = session_id
                    st.rerun()

    # --- Main Chat Interface ---
    st.title("🤖 Multimodal RAG LLM Chatbot")
    st.caption("An AI assistant for the Inhouse Digital Marketing Team")

    # Define a unique key for the chat history object based on the current session ID
    history_key = f"history_{st.session_state.session_id}"
    chat_history = StreamlitChatMessageHistory(key=history_key)

    # Load history from S3 if the session is new and the history is empty
    if not chat_history.messages:
        chat_history.messages = load_chat_history(st.session_state.session_id)

    # Display previous chat messages
    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    # Handle user input
    if user_input := st.chat_input("Ask about your active ad campaigns..."):
        st.chat_message("human").write(user_input)
        
        with st.spinner("Analyzing data and generating response..."):
            # --- Modify input with language instruction ---
            # Get the language from the sidebar's selectbox
            final_input = f"Please answer the following question in {output_language}: {user_input}"
            # ---------------------------------------------

            # Instantiate the RAGChain with the current session's chat history
            rag_chain = RAGChain(chat_history)
            
            # Run the chain to get a response
            response = rag_chain.run(final_input)
            
            st.chat_message("ai").write(response)
            
            # Save the updated chat history to S3
            save_chat_history(st.session_state.session_id, chat_history.messages)

# --- Application Gatekeeper ---
# The main app will only run if the password check passes.
if check_password():
    run_app()