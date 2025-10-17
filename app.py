# app.py

import streamlit as st
import streamlit_authenticator as stauth
import yaml
import uuid
from yaml.loader import SafeLoader
from src.openai_chain import OpenAIRAGChain # We only need the RAG chain now
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from src.utils import (
    save_chat_history,
    load_chat_history,
    get_saved_sessions,
    get_chat_title
)

st.set_page_config(page_title="Ads Manager AI Assistant", layout="wide")

# --- 1. USER AUTHENTICATION ---
try:
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("🚨 Config file not found! Please make sure 'config.yaml' is in the root directory.")
    st.stop()


authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render the login module
authenticator.login()


# --- 2. CACHED RESOURCES ---
@st.cache_resource
def load_rag_chain(_chat_memory):
    """
    Load the RAG chain. This is cached for performance.
    It assumes the vector database is already populated.
    """
    return OpenAIRAGChain(_chat_memory)


# --- 3. MAIN CHATBOT APPLICATION ---
def run_chatbot_app():
    """
    This function runs the main chatbot interface after a user is logged in.
    """
    # Get the logged-in username
    username = st.session_state.get("username")
    if not username:
        st.warning("Could not determine user. Please log in again.")
        st.stop()

    # --- Sidebar for Conversations ---
    with st.sidebar:
        st.title(f"Welcome, *{st.session_state['name']}*")
        st.markdown("---")
        
        if st.button("➕ New Chat"):
            # Clear session-specific state to start a fresh chat
            st.session_state.session_id = str(uuid.uuid4())
            history_key = f"history_{st.session_state.session_id}"
            if history_key in st.session_state:
                del st.session_state[history_key]
            st.rerun()

        st.markdown("## Your Conversations")
        
        # Fetch and display saved sessions for the current user
        saved_sessions = get_saved_sessions(username)
        for session_id in saved_sessions:
            title = get_chat_title(username, session_id)
            if st.button(title, key=session_id, use_container_width=True):
                # Load the selected chat session
                st.session_state.session_id = session_id
                st.rerun()
        
        st.markdown("---")
        authenticator.logout("Logout", "sidebar")


    # --- Session Initialization ---
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Unique key for chat history based on session_id
    history_key = f"history_{st.session_state.session_id}"
    chat_history = StreamlitChatMessageHistory(key=history_key)

    # --- Load History & Initialize Chain ---
    # Load history from cloud if the session is new
    if not chat_history.messages:
        chat_history.messages = load_chat_history(username, st.session_state.session_id)

    # Initialize the RAG chain
    llm_chain = load_rag_chain(chat_history)

    # --- Main Chat Interface ---
    st.title("📈 Ads Manager AI Assistant")
    st.caption("Ask me anything about your current ad campaigns.")

    chat_container = st.container()

    # Display previous chat messages
    with chat_container:
        for msg in chat_history.messages:
            st.chat_message(msg.type).write(msg.content)

    # Handle user input
    if user_input := st.chat_input("Ask about your campaigns, ad sets, or performance..."):
        # Add user message to history and display it
        chat_history.add_user_message(user_input)
        with chat_container:
            st.chat_message("user").write(user_input)

        # Get AI response
        with st.spinner("Analyzing data..."):
            llm_response = llm_chain.run(user_input=user_input)
            
        # Add AI response to history and display it
        chat_history.add_ai_message(llm_response)
        with chat_container:
            st.chat_message("ai").write(llm_response)

        # Save the entire updated conversation to the cloud
        save_chat_history(username, st.session_state.session_id, chat_history.messages)

        # We don't need a full rerun here as Streamlit handles the chat element updates
        # A rerun would clear the input box, which is desirable for some UX flows.
        # st.rerun()


# --- GATEKEEPER ---
# This controls what the user sees based on their login status.

if st.session_state["authentication_status"]:
    run_chatbot_app()
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')