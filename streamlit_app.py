import streamlit as st
import requests
import time
import re

# Flask API base URL
API_BASE_URL = "http://localhost:5000/api"

# Set up Streamlit page config
st.set_page_config(
    page_title="Raggle",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for main app
st.markdown("""
<style>
    .main {
        padding: 2rem;
        background-color: #1e1e1e;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #1c6758;
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #134e4a;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .stChatInput > div > div > input {
        border-radius: 7px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        max-width: 80%;
    }
    .chat-message.user {
        background-color: #dcf8c6;
        color: black;
        margin-left: auto;
        flex-direction: row-reverse; /* Reverse the order: message, then avatar */
    }
    .chat-message.bot {
        background-color: #e6f7ff;
        color: black;
    }
    .chat-message .avatar {
        width: 35px;
        margin-left: 1rem; /* For user messages (reversed), this puts space between message and avatar */
        margin-right: 1rem; /* For bot messages, this puts space between avatar and message */
    }
    .chat-message.user .avatar {
        margin-left: 1rem; /* Ensure spacing between message and avatar for user */
        margin-right: 0; /* No margin on the right for user avatar */
    }
    .chat-message.bot .avatar {
        margin-left: 0; /* No margin on the left for bot avatar */
        margin-right: 1rem; /* Space between bot avatar and message */
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .chat-time {
        font-size: 0.8em;
        color: gray;
        margin-left: 1rem; /* Space between message and timestamp */
    }
    .chat-message.user .chat-time {
        margin-left: 0;
        margin-right: 1rem; /* For user messages, timestamp is on the left of the message */
    }
    .document-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f5f5f5;
        color: black;
        margin: 0.5rem 0;
    }
    .chat-container {
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #333;
        border-radius: 10px;
        background-color: #1e1e1e;
    }
    .header h1 {
        color: white;
    }
    .stSidebar {
        background-color: #2a2a2a;
    }
    .stSidebar h2, .stSidebar h3 {
        color: white;
    }
    .stSidebar p, .stSidebar small {
        color: #d3d3d3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'stats' not in st.session_state:
    st.session_state.stats = {"vector_count": 0}

# Chat response formatting
def format_assistant_response(response):
    # Ensure the response is a string and strip any existing HTML tags to prevent double formatting
    response = str(response).strip()
    response = re.sub(r'<[^>]+>', '', response)
    # Remove any markdown-style bold markers
    response = re.sub(r'\*\*?(.*?)\*\*?', r'\1', response)
    # Split into lines and format
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if not lines:
        return "No response generated."
    formatted_response = "<div style='line-height: 1.5;'>"
    formatted_response += "<b>ANSWER:</b><br>"
    for line in lines:
        if line.isupper() or line.endswith(':'):
            formatted_response += f"<b>{line}</b><br>"
        else:
            formatted_response += f"{line}<br>"
    formatted_response += "</div>"
    return formatted_response

# Function to display chat messages with consistent formatting and styling
def display_chat_message(role, content, timestamp):
    # Apply formatting for assistant messages
    formatted_content = format_assistant_response(content) if role == "assistant" else content
    # Render the message with the appropriate styling
    if role == "user":
        st.markdown(
            f'<div class="chat-message user">'
            f'<div class="message">{formatted_content}</div>'
            f'<div class="chat-time">{timestamp}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-message bot">'
            f'<div class="message">{formatted_content}</div>'
            f'<div class="chat-time">{timestamp}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# Fetch initial data
def fetch_initial_data():
    if not st.session_state.documents:
        doc_response = requests.get(f"{API_BASE_URL}/documents")
        if doc_response.status_code == 200:
            st.session_state.documents = doc_response.json()["documents"]
    if not st.session_state.stats["vector_count"]:
        stats_response = requests.get(f"{API_BASE_URL}/stats")
        if stats_response.status_code == 200:
            st.session_state.stats = stats_response.json()

# Main app
def main():
    st.markdown('<div class="header"><h1>🤖Raggle Assistant📑</h1></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<h2>📑Document Upload</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "csv", "txt", "png", "jpg", "jpeg", "mp4"])

        def process_document():
            if uploaded_file:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.documents = data["documents"]
                        st.success(data["message"])
                        fetch_initial_data()
                    else:
                        st.error(response.json().get("error", "Unknown error"))

        if uploaded_file:
            st.button("Process Document", key="process_btn", on_click=process_document)
        
        st.markdown('<h3>YouTube Video Processing</h3>', unsafe_allow_html=True)
        youtube_url = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

        def process_youtube():
            if youtube_url:
                # Validate URL format first
                if not youtube_url.strip() or ("youtube.com" not in youtube_url and "youtu.be" not in youtube_url):
                    st.error("Please enter a valid YouTube URL")
                    return
                # Clean up the URL if it's a short youtu.be link
                clean_url = youtube_url
                if "youtu.be" in youtube_url:
                    video_id = youtube_url.split('/')[-1].split('?')[0]
                    clean_url = f"https://www.youtube.com/watch?v={video_id}"
                    
                with st.spinner("Processing YouTube video..."):
                    try:
                        response = requests.post(f"{API_BASE_URL}/upload", data={"youtube_url": clean_url})
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.documents = data["documents"]
                            st.success(data["message"])
                            fetch_initial_data()
                        else:
                            st.error(response.json().get("error", "Unknown error"))
                    except Exception as e:
                        st.error(f"Error processing YouTube video: {str(e)}")
            else:
                st.error("Please enter a YouTube URL first")

        if youtube_url:
            st.button("Process YouTube Video", key="process_youtube_btn", on_click=process_youtube)

        fetch_initial_data()
        if st.session_state.documents:
            st.markdown("<h3>Processed Documents</h3>", unsafe_allow_html=True)
            for doc in st.session_state.documents:
                st.markdown(
                    f'<div class="document-card">'
                    f'<strong>{doc["name"]}</strong><br>'
                    f'<small>Added: {doc["time"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            f'<div class="db-stats">'
            f'<h3>Database Stats</h3>'
            f'<p>Vector Count: {st.session_state.stats["vector_count"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        def clear_database():
            response = requests.post(f"{API_BASE_URL}/clear")
            if response.status_code == 200:
                st.session_state.documents = []
                st.session_state.stats = {"vector_count": 0}
                st.success(response.json()["message"])

        st.button("Clear Database", key="clear_db", on_click=clear_database)

        st.markdown("<h3>Model Settings</h3>", unsafe_allow_html=True)
        config_response = requests.get(f"{API_BASE_URL}/config")
        if config_response.status_code == 200 and config_response.json()["gemini_api_key_set"]:
            st.success("Gemini API key is set")
        else:
            st.error("Gemini API key is not set. Please add it to your .env file.")

    col1, col2 = st.columns([8, 1])
    with col1:
        # Chat display area with st.chat_message
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    display_chat_message(message["role"], message["content"], message["time"])

        # Function to handle chat processing
        def process_chat_input(user_input):
            timestamp = time.strftime("%H:%M:%S")
            
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    display_chat_message("user", user_input, timestamp)
            
            # Add user message to history (store raw content)
            st.session_state.chat_history.append({"role": "user", "content": user_input, "time": timestamp})

            # Process assistant response
            if not st.session_state.documents:
                raw_response = (
                    "Hello!\n"
                    "It seems no documents or YouTube links have been uploaded yet. "
                    "Please upload a document or provide a YouTube URL in the sidebar to start chatting with your content. "
                    "I'm excited to assist you once your data is ready!"
                )
                with chat_container:
                    with st.chat_message("assistant"):
                        display_chat_message("assistant", raw_response, timestamp)
                # Store the raw response in history
                st.session_state.chat_history.append({"role": "assistant", "content": raw_response, "time": timestamp})
            else:
                with st.spinner("Generating response..."):
                    response = requests.post(f"{API_BASE_URL}/chat", json={"query": user_input})
                    if response.status_code == 200:
                        data = response.json()
                        raw_response = data["response"]
                        with chat_container:
                            with st.chat_message("assistant"):
                                display_chat_message("assistant", raw_response, data["timestamp"])
                        # Store the raw response in history
                        st.session_state.chat_history.append({"role": "assistant", "content": raw_response, "time": data["timestamp"]})
                    else:
                        st.error("Error generating response")

        # Chat input with submit logic
        chat_input = st.chat_input("Ask something about your documents...", key="chat_input")
        if chat_input:
            process_chat_input(chat_input)

        def clear_chat():
            st.session_state.chat_history = []

        st.button("Clear Chat History", key="clear_chat", on_click=clear_chat)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<div style="text-align: center; color: #888888; padding: 10px;">'
            '<p>© 2025 RAG Chatbot. All Rights Reserved.</p>'
            '<p>Developed by: <b>Jagdeesh P</b></p>'
            '<p style="font-size: 0.8em;">Version 1.2.5 | Built with Streamlit</p>'
            '</div>',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()