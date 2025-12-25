import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models import ChatHuggingFace


from datetime import datetime
import json
import os
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize HuggingFace model
def get_model():
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )
    return ChatHuggingFace(llm=llm)

# Page configuration
st.set_page_config(
    page_title="Study Buddy - Your AI Homework Hero",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI matching React design
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #fce4ec 0%, #e1bee7 50%, #c5cae9 100%);
    }
    
    /* Sidebar styling - Light background with proper colors */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(252, 228, 236, 0.9) 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #333 !important;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #7b1fa2 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #333 !important;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #ec407a 0%, #ab47bc 100%);
        color: white !important;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        border-bottom-right-radius: 5px;
        margin: 0.75rem 0;
        margin-left: 20%;
        box-shadow: 0 3px 8px rgba(236, 64, 122, 0.3);
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        color: #1a1a1a !important;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        border-bottom-left-radius: 5px;
        margin: 0.75rem 0;
        margin-right: 20%;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .timestamp {
        font-size: 0.7rem;
        opacity: 0.75;
        margin-top: 0.5rem;
    }
    
    /* File upload area - Light background */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        border: 2px dashed #e1bee7 !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #7b1fa2 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
        color: #666 !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #666 !important;
    }
    
    .uploaded-file {
        background: linear-gradient(135deg, #e1bee7 0%, #f8bbd0 100%);
        padding: 0.5rem 1rem;
        border-radius: 12px;
        margin: 0.25rem;
        display: inline-block;
        font-size: 0.85rem;
        color: #333 !important;
        font-weight: 500;
    }
    
    /* Input area - White background with dark text */
    .stTextArea textarea {
        border-radius: 18px !important;
        border: 2px solid #e1bee7 !important;
        background: white !important;
        color: #1a1a1a !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #999 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #ab47bc !important;
        box-shadow: 0 0 0 3px rgba(171, 71, 188, 0.15) !important;
    }
    
    .stTextArea label {
        color: #999 !important;
        font-size: 0.9rem !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #ec407a 0%, #ab47bc 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s !important;
        box-shadow: 0 3px 8px rgba(171, 71, 188, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 5px 15px rgba(171, 71, 188, 0.4) !important;
    }
    
    /* Text input styling */
    .stTextInput input {
        background: white !important;
        color: #1a1a1a !important;
        border: 2px solid #e1bee7 !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #ab47bc !important;
        box-shadow: 0 0 0 3px rgba(171, 71, 188, 0.15) !important;
    }
    
    .stTextInput label {
        color: #7b1fa2 !important;
        font-weight: 600 !important;
    }
    
    /* Subject badge */
    .subject-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ab47bc 0%, #ec407a 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 3px 8px rgba(171, 71, 188, 0.3);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            'role': 'assistant',
            'content': "Hey bestie! 💖 I'm here to help you crush your homework. Pick a subject and let's get started!",
            'timestamp': datetime.now()
        }
    ]

if 'selected_subject' not in st.session_state:
    st.session_state.selected_subject = None

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Subjects with exact colors from React design
subjects = [
    {'id': 'math', 'name': 'Math', 'icon': '📐', 'gradient': 'linear-gradient(135deg, #ec407a 0%, #f06292 100%)'},
    {'id': 'science', 'name': 'Science', 'icon': '🔬', 'gradient': 'linear-gradient(135deg, #ab47bc 0%, #7e57c2 100%)'},
    {'id': 'english', 'name': 'English', 'icon': '📖', 'gradient': 'linear-gradient(135deg, #42a5f5 0%, #26c6da 100%)'},
    {'id': 'cs', 'name': 'Computer Science', 'icon': '💻', 'gradient': 'linear-gradient(135deg, #7e57c2 0%, #ab47bc 100%)'},
    {'id': 'nepali', 'name': 'Nepali', 'icon': '🇳🇵', 'gradient': 'linear-gradient(135deg, #ff5722 0%, #ff6f00 100%)'},
    {'id': 'social', 'name': 'Social', 'icon': '🌍', 'gradient': 'linear-gradient(135deg, #26a69a 0%, #00897b 100%)'},
    {'id': 'health', 'name': 'Health', 'icon': '💊', 'gradient': 'linear-gradient(135deg, #26c6da 0%, #00acc1 100%)'},
]

# Sidebar
with st.sidebar:
    st.markdown("## 📚 Pick Subject")
    
    # Subject buttons with exact styling
    cols = st.columns(2)
    for idx, subject in enumerate(subjects):
        with cols[idx % 2]:
            button_style = f"""
                background: {subject['gradient']};
                color: white;
                border: none;
                border-radius: 12px;
                padding: 1rem 0.5rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15);
                text-align: center;
                width: 100%;
            """ if st.session_state.selected_subject == subject['name'] else """
                background: rgba(255, 255, 255, 0.7);
                color: #666;
                border: 2px solid rgba(225, 190, 231, 0.5);
                border-radius: 12px;
                padding: 1rem 0.5rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                text-align: center;
                width: 100%;
            """
            
            if st.button(
                f"{subject['icon']}\n{subject['name']}", 
                key=f"subject_{subject['id']}",
                use_container_width=True
            ):
                st.session_state.selected_subject = subject['name']
                st.rerun()
    
    st.markdown("---")
    
    # File upload section
    st.markdown("## 📎 Upload Files")
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=['pdf', 'txt', 'docx', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        label_visibility="visible",
        help="Limit 200MB per file • PDF, TXT, DOCX, JPG, JPEG, PNG"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
    
    st.markdown("---")
    
    # Link input
    st.markdown("## 🔗 Add YouTube/Website URL")
    link_input = st.text_input("", placeholder="https://...", label_visibility="collapsed")
    if st.button("Add Link", use_container_width=True) and link_input:
        if 'links' not in st.session_state:
            st.session_state.links = []
        st.session_state.links.append(link_input)
        st.rerun()
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True, type="primary"):
        st.session_state.messages = [
            {
                'role': 'assistant',
                'content': "Hey bestie! 💖 I'm here to help you crush your homework. Pick a subject and let's get started!",
                'timestamp': datetime.now()
            }
        ]
        st.session_state.uploaded_files = []
        if 'links' in st.session_state:
            st.session_state.links = []
        st.rerun()

# Main content
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; background: linear-gradient(135deg, #ec407a 0%, #ab47bc 50%, #5c6bc0 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               font-size: 2rem; font-weight: 800;">
        Study Buddy ✨
    </h1>
    <p style="margin:0; color: #666; font-size: 0.9rem; margin-top: 0.25rem;">Your AI homework hero</p>
</div>
""", unsafe_allow_html=True)

# Show selected subject
if st.session_state.selected_subject:
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <span class="subject-badge">
            ✨ Studying: {st.session_state.selected_subject}
        </span>
    </div>
    """, unsafe_allow_html=True)

# Display uploaded files
if st.session_state.uploaded_files:
    st.markdown("**📎 Uploaded Files:**")
    file_html = ""
    for file in st.session_state.uploaded_files:
        file_html += f'<span class="uploaded-file">📄 {file.name}</span> '
    st.markdown(f'<div style="margin-bottom: 1rem;">{file_html}</div>', unsafe_allow_html=True)

# Display links
if 'links' in st.session_state and st.session_state.links:
    st.markdown("**🔗 Links:**")
    link_html = ""
    for link in st.session_state.links:
        link_html += f'<span class="uploaded-file">🔗 {link[:50]}...</span> '
    st.markdown(f'<div style="margin-bottom: 1rem;">{link_html}</div>', unsafe_allow_html=True)

st.markdown("---")

# Chat messages container
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                {message['content']}
                <div class="timestamp">{message['timestamp'].strftime('%I:%M %p')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                {message['content']}
                <div class="timestamp">{message['timestamp'].strftime('%I:%M %p')}</div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
st.markdown("---")
user_input = st.text_area(
    "Ask me anything... Type your question here! 💭",
    key="user_input",
    height=120,
    placeholder="Press Ctrl+Enter to send"
)

col1, col2, col3 = st.columns([6, 1, 1])

with col2:
    send_button = st.button("📤 Send", use_container_width=True, type="primary")

# Handle send
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now()
    })
    
    # Prepare context
    context = f"Subject: {st.session_state.selected_subject or 'General'}\n"
    context += f"User Question: {user_input}\n"
    
    # Add file context
    if st.session_state.uploaded_files:
        context += f"\nFiles uploaded: {len(st.session_state.uploaded_files)} files\n"
        for file in st.session_state.uploaded_files:
            if file.type.startswith('image'):
                try:
                    image = Image.open(file)
                    context += f"Image file: {file.name}\n"
                except:
                    pass
            else:
                try:
                    content = file.read().decode('utf-8', errors='ignore')[:1000]
                    context += f"\nContent from {file.name}:\n{content}\n"
                except:
                    pass
    
    # Add link context
    if 'links' in st.session_state and st.session_state.links:
        context += f"\nLinks provided: {', '.join(st.session_state.links)}\n"
    
    # System prompt
    system_prompt = """You are StudyBuddy, a friendly and enthusiastic AI tutor. 
    You help students with their homework in a fun, supportive way. 
    - Be encouraging and positive
    - Break down complex topics into easy steps
    - Use examples and analogies
    - Keep responses concise but thorough
    - Use emojis occasionally to be friendly
    - If you don't have enough context, ask clarifying questions"""
    
    full_prompt = f"{system_prompt}\n\n{context}"
    
    try:
        # Generate response with HuggingFace
        with st.spinner("Thinking... 🤔"):
            model = get_model()
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = model.invoke(messages)
            assistant_response = response.content
        
        # Add assistant message
        st.session_state.messages.append({
            'role': 'assistant',
            'content': assistant_response,
            'timestamp': datetime.now()
        })
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.messages.append({
            'role': 'assistant',
            'content': "Sorry bestie! I had a little hiccup. Can you try asking again? 💖",
            'timestamp': datetime.now()
        })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.85rem; padding: 1rem;">
    Made with 💖 by Study Buddy | Powered by HuggingFace Zephyr-7B
</div>
""", unsafe_allow_html=True)