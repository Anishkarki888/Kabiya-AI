from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

st.set_page_config(page_title="Kabiya's Baddie AI", page_icon="💅")

def main():
    # Fun header with baddie vibes
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B8B, #FFE569, #FF8E8E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
    }
    .baddie-vibes {
        text-align: center;
        color: #FF6B8B;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">💋 Kabiya\'s Baddie AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="baddie-vibes">💅 Too hot to handle, too cool to care • She slays, AI obeys</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for baddie settings
    with st.sidebar:
        st.header("🎛️ Baddie Settings")
        attitude_level = st.slider("Attitude Level", 0.1, 1.0, 0.8, 0.1, help="How much sass in the response?")
        response_length = st.slider("Response Length", 50, 200, 80, 10, help="Keep it short and sassy!")
        
        st.markdown("---")
        st.info("""
        **Baddie Mode Guide:**
        - Low (0.1-0.3): Chill vibes only 🥱
        - Medium (0.4-0.7): Balanced sass 💁‍♀️  
        - High (0.8-1.0): Maximum slayage 🔥
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Queen Kabiya's AI")
        
        # Example prompts with baddie themes
        example_prompts = [
            "How to slay like Kabiya?",
            "Best comeback for haters?",
            "Quick confidence boost?",
            "Main character energy tips"
        ]
        
        selected_prompt = st.selectbox("Or choose a baddie topic:", [""] + example_prompts)
        
        if selected_prompt:
            user_input = st.text_area("Your question to the queen:", value=selected_prompt, height=80)
        else:
            user_input = st.text_area("Your question to the queen:", placeholder="Ask about fashion, confidence, or how to be iconic...", height=80)
    
    with col2:
        st.subheader("👑 Baddie Rules")
        st.markdown("""
        - **Confidence is key** 🔑
        - **Own your slay** 💃
        - **Haters = Motivation** 📈
        """)
    
    st.markdown("---")
    
    if st.button("💋 Get Baddie Advice", type="primary"):
        if not user_input.strip():
            st.warning("👀 Queen needs a question, bestie!")
            return
        
        with st.spinner("💅 Slaying your question..."):
            try:
                # Use a reliable model
                repo_id = "HuggingFaceH4/zephyr-7b-beta"  
                
                llm = HuggingFaceEndpoint(
                    repo_id=repo_id,
                    max_new_tokens=response_length,  # Shorter responses
                    temperature=attitude_level,
                    top_p=0.9,
                )
                
                model = ChatHuggingFace(llm=llm)
                
                # SHORT and sassy prompt
                baddie_prompt = f"""You are Kabiya's sassy AI - a confident baddie who gives SHORT, ICONIC advice.

QUESTION: {user_input}

Respond in 3-4 lines MAX with:
- Sassy, confident energy
- Quick practical tips
- Fire emojis
- Keep it SHORT and iconic

SHORT ANSWER:"""
                
                messages = [HumanMessage(content=baddie_prompt)]
                response = model.invoke(messages)
                
                # Display the short response
                st.subheader("👑 Queen K Says:")
                st.success(response.content)
                    
            except Exception as e:
                st.error(f"❌ Oops! Bad day: {str(e)}")

    # Footer
    st.markdown("---")
    st.caption("Made for Queen Kabiya • Short answers only • Stay iconic! ✨")

if __name__ == "__main__":
    main()