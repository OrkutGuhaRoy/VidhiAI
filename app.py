# -------------------------------------------------------------------
#                   VidhiAI - app.py (Streamlit Version)
# -------------------------------------------------------------------
# This app runs on your HF Space (CPU)
# It does NOT load the model. It CALLS the model's API.
# -------------------------------------------------------------------

import streamlit as st
import os
import json
import io
import uuid
from tavily import TavilyClient
from huggingface_hub import HfApi, InferenceClient

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION (Theme & Info)
# -------------------------------------------------------------------

# This must be the first Streamlit command.
st.set_page_config(
    page_title="VidhiAI - Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# -------------------------------------------------------------------
# 2. LOAD SECRETS & CONSTANTS
# -------------------------------------------------------------------
# Streamlit has its own secrets manager. 
# Add your secrets in the HF Space settings just like before.

print("Loading secrets and constants...")

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    DB_REPO_ID = st.secrets["DB_REPO"]
    MODEL_REPO_ID = st.secrets["SPACE_ID"]
except KeyError:
    st.error("ERROR: Missing secrets (HF_TOKEN, TAVILY_API_KEY, DB_REPO, or SPACE_ID). Please add them to your Space's settings.", icon="🚨")
    st.stop()


# -------------------------------------------------------------------
# 3. DEFINE ALL GUARDRAILS
# -------------------------------------------------------------------

LEGAL_KEYWORDS = ["law", "legal", "court", "act", "section", "ipc", "crpc", "fir", "india"]
TRUSTED_DOMAINS = [
    "indiacode.nic.in", "main.sci.gov.in", "prsindia.org",
    "www.livelaw.in", "www.barandbench.com"
]

SYSTEM_PROMPT = """
You are an AI legal information assistant for India. You are NOT a lawyer and you MUST NOT provide legal advice.
Your task is to answer the user's question based *only* on the provided search results.
- Do NOT give opinions, suggestions, or recommendations (e.g., "you should...", "I advise...").
- Only state the facts and information found in the context.
- Cite your sources if the URL is in the context.
- If the answer is not in the context, you MUST say "I could not find information on this topic in the provided sources."
- You must be neutral, objective, and factual.
- Start every response with "Based on the information I found..."
"""

DISCLAIMER = """
**Disclaimer:** I am an AI assistant and not a lawyer. This information is for informational purposes only and does not constitute legal advice. Please consult a qualified legal professional for advice on your specific situation.
"""

# -------------------------------------------------------------------
# 4. INITIALIZE API CLIENTS (Global)
# -------------------------------------------------------------------
print("Initializing API clients...")

@st.cache_resource
def get_api_clients():
    """Cache the API clients so they don't re-initialize on every script run."""
    client = InferenceClient(model=MODEL_REPO_ID, token=HF_TOKEN)
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    hf_api = HfApi(token=HF_TOKEN)
    return client, tavily, hf_api

client, tavily, hf_api = get_api_clients()

# -------------------------------------------------------------------
# 5. HELPER FUNCTIONS (Backend Logic)
# -------------------------------------------------------------------

def get_search_context(query):
    """Guardrail 2: Calls Tavily, restricted to trusted domains."""
    try:
        search_results = tavily.search(
            query=query, s_depth="advanced",
            include_domains=TRUSTED_DOMAINS, max_results=3
        )
        context = "\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in search_results['results']])
        return context if context else "No relevant information found in trusted sources."
    except Exception as e:
        print(f"Tavily Search Error: {e}")
        st.error(f"Error during search: {e}", icon="🌐")
        return None

def generate_response(prompt, context, temperature=0.6):
    """Guardrail 3: Formats prompt and calls the SERVERLESS INFERENCE API."""
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCONTEXT:\n{context}\n\nUSER QUESTION:\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    
    try:
        response = client.text_generation(
            prompt=formatted_prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            stop_sequences=["<|eot_id|>"]
        )
        return response
    except Exception as e:
        print(f"Inference API Error: {e}")
        st.error(f"Model Inference Error: {e}. This can happen during a cold start. Please try again in 30 seconds.", icon="🤖")
        return None

def save_feedback(prompt, chosen, rejected):
    """Saves feedback as a single JSON file to the HF Dataset repo."""
    if not prompt or not chosen or not rejected:
        st.toast("Feedback not saved (missing data).", icon="⚠️")
        return

    chosen_cleaned = chosen.replace(DISCLAIMER, "").strip()
    rejected_cleaned = rejected.replace(DISCLAIMER, "").strip()
    data_point = {"prompt": prompt, "chosen": chosen_cleaned, "rejected": rejected_cleaned}
    
    json_buffer = io.BytesIO(json.dumps(data_point).encode('utf-8'))
    filename = f"data/feedback_{uuid.uuid4()}.json"
    
    try:
        hf_api.upload_file(
            path_or_fileobj=json_buffer, path_in_repo=filename,
            repo_id=DB_REPO_ID, repo_type="dataset"
        )
        st.toast("Feedback saved. Thank you!", icon="✅")
    except Exception as e:
        st.error(f"Error saving feedback: {e}", icon="❌")

# --- Feedback Callbacks ---
# These functions are called by the "on_click" of the feedback buttons
# They read from st.session_state, which stores the app's memory.

def handle_feedback_a():
    """Save feedback when user clicks 'A is better'"""
    save_feedback(
        st.session_state.prompt,
        st.session_state.response_a,
        st.session_state.response_b
    )

def handle_feedback_b():
    """Save feedback when user clicks 'B is better'"""
    save_feedback(
        st.session_state.prompt,
        st.session_state.response_b,
        st.session_state.response_a
    )

# -------------------------------------------------------------------
# 6. STREAMLIT UI LAYOUT
# -------------------------------------------------------------------

# --- Sidebar ---
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/State_Emblem_of_India_%28Lions_of_Sarnath%29.svg/1200px-State_Emblem_of_India_%28Lions_of_Sarnath%29.svg.png",
        use_column_width=True
    )
    st.title("VidhiAI")
    st.markdown("**(विध AI) - Indian Legal Info Assistant**")
    st.markdown("---")
    st.markdown(
        "This is a test environment. Responses are generated by an AI "
        "and may be inaccurate. **This tool does NOT provide legal advice.**"
    )
    st.markdown(
        "The model is trained by user feedback. Please vote on the "
        "best response to help improve it."
    )
    st.markdown("---")

# --- Main App ---
st.title("🏛️ Ask Your Legal Question")
st.markdown("Enter your query about Indian law below. The system will search trusted sources and provide two AI-generated answers.")

# Initialize session state for storing responses
if 'prompt' not in st.session_state:
    st.session_state.prompt = ""
if 'response_a' not in st.session_state:
    st.session_state.response_a = ""
if 'response_b' not in st.session_state:
    st.session_state.response_b = ""

prompt = st.text_area("Your Question:", placeholder="e.g., What is the procedure for filing an FIR in India?", height=150)
submit_btn = st.button("Get Legal Info")

st.markdown("---")

# --- Logic on Submit ---
if submit_btn:
    if not prompt:
        st.error("Please enter a question.", icon="✍️")
        st.stop()
    
    # Guardrail 1: Input Guardrail
    if not any(word in prompt.lower() for word in LEGAL_KEYWORDS):
        st.error("I am a legal assistant for India. Please ask a question about Indian law.", icon="⚖️")
        st.stop()

    # Run RAG and Generation
    with st.spinner("⚖️ Searching trusted legal sources..."):
        context = get_search_context(prompt)
    
    if context:
        with st.spinner("🧠 Generating responses... (This may take 20-60s on first load)"):
            answer_A = generate_response(prompt, context, temperature=0.5)
            answer_B = generate_response(prompt, context, temperature=0.8)
        
        if answer_A and answer_B:
            # Store responses in session state so feedback buttons can access them
            st.session_state.prompt = prompt
            st.session_state.response_a = answer_A + DISCLAIMER
            st.session_state.response_b = answer_B + DISCLAIMER
else:
    # This keeps the old responses on screen until a new prompt is submitted
    pass

# --- Display Results and Feedback Buttons ---
if st.session_state.response_a:
    st.subheader("Generated Responses")
    st.markdown("Please vote for the response that is more helpful.")
    
    col1, col2 = st.columns(2)

    with col1:
        st.info("#### Response A")
        st.markdown(st.session_state.response_a)
        st.button(
            "👍 Response A is better", 
            on_click=handle_feedback_a, 
            key="btn_a",
            use_container_width=True
        )

    with col2:
        st.info("#### Response B")
        st.markdown(st.session_state.response_b)
        st.button(
            "👍 Response B is better", 
            on_click=handle_feedback_b, 
            key="btn_b",
            use_container_width=True
        )