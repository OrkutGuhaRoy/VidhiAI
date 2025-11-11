# -------------------------------------------------------------------
#                   VidhiAI - app.py (CPU + 4-bit quantized)
# -------------------------------------------------------------------
import streamlit as st
import os
import json
import io
import uuid
import time
from tavily import TavilyClient
from huggingface_hub import HfApi
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM , pipeline
from requests.exceptions import RequestException

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="VidhiAI - Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# -------------------------------------------------------------------
# 2. LOAD SECRETS & CONSTANTS
# -------------------------------------------------------------------
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    DB_REPO_ID = st.secrets["DB_REPO"]
    MODEL_REPO_ID = st.secrets["SPACE_ID"]
except KeyError:
    st.error("Missing secrets (HF_TOKEN, TAVILY_API_KEY, DB_REPO, or SPACE_ID).", icon="🚨")
    st.stop()

LEGAL_KEYWORDS = ["law", "legal", "court", "act", "section", "ipc", "crpc", "fir", "india"]
TRUSTED_DOMAINS = [
    "indiacode.nic.in", "main.sci.gov.in", "prsindia.org",
    "www.livelaw.in", "www.barandbench.com"
]

SYSTEM_PROMPT = """
You are an AI legal information assistant for India. You are NOT a lawyer and you MUST NOT provide legal advice.
Answer user's questions based only on the provided search results.
- Do NOT give opinions or recommendations.
- Cite sources if URLs are in context.
- If answer not in context, say "I could not find information on this topic in the provided sources."
- Be neutral, objective, factual.
- Start every response with "Based on the information I found..."
"""

DISCLAIMER = """
**Disclaimer:** I am an AI assistant and not a lawyer. This information is for informational purposes only and does not constitute legal advice. Please consult a qualified legal professional for advice on your specific situation.
"""

# -------------------------------------------------------------------
# 3. LOAD MODEL (CPU + 4-bit)
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    st.info("Loading model locally... First run may take 30–60s.", icon="🧠")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO_ID,
        device_map="auto",   # CPU
        load_in_4bit=True    # 4-bit quantization to reduce RAM
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    hf_api = HfApi(token=HF_TOKEN)
    return pipe, tavily, hf_api

# -------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# -------------------------------------------------------------------
def get_search_context(query, tavily_client):
    try:
        results = tavily_client.search(
            query=query,
            s_depth="advanced",
            include_domains=TRUSTED_DOMAINS,
            max_results=3
        ).get("results") or []
        context = "\n".join([f"Source: {r.get('url')}\nContent: {r.get('content')}" for r in results])
        return context if context else "No relevant information found in trusted sources."
    except Exception as e:
        st.error(f"Tavily Search Error: {e}", icon="🌐")
        return None

def generate_response(pipe, prompt, context, temperature=0.6):
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}
<|eot_id|><|start_header_id|>user<|end_header_id|>
CONTEXT:
{context}

USER QUESTION:
{prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    try:
        out = pipe(formatted_prompt, max_new_tokens=512, temperature=temperature, do_sample=True)[0]["generated_text"]
        if "<|start_header_id|>assistant<|end_header_id|>" in out:
            out = out.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return out
    except Exception as e:
        st.error(f"Generation error: {e}", icon="🤖")
        return None

def save_feedback(prompt, chosen, rejected, hf_api):
    if not prompt or not chosen or not rejected:
        st.toast("Feedback not saved (missing data).", icon="⚠️")
        return
    data_point = {
        "prompt": prompt,
        "chosen": chosen.replace(DISCLAIMER, "").strip(),
        "rejected": rejected.replace(DISCLAIMER, "").strip()
    }
    json_buffer = io.BytesIO(json.dumps(data_point).encode("utf-8"))
    filename = f"data/feedback_{uuid.uuid4()}.json"
    try:
        hf_api.upload_file(path_or_fileobj=json_buffer, path_in_repo=filename, repo_id=DB_REPO_ID, repo_type="dataset")
        st.toast("Feedback saved. Thank you!", icon="✅")
    except Exception as e:
        st.error(f"Error saving feedback: {e}", icon="❌")

def handle_feedback_a():
    pipe, tavily, hf_api = st.session_state["_clients"]
    save_feedback(st.session_state.prompt, st.session_state.response_a, st.session_state.response_b, hf_api)

def handle_feedback_b():
    pipe, tavily, hf_api = st.session_state["_clients"]
    save_feedback(st.session_state.prompt, st.session_state.response_b, st.session_state.response_a, hf_api)

# -------------------------------------------------------------------
# 5. STREAMLIT UI
# -------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/State_Emblem_of_India_%28Lions_of_Sarnath%29.svg/1200px-State_Emblem_of_India_%28Lions_of_Sarnath%29.svg.png",
        use_column_width=True
    )
    st.title("VidhiAI")
    st.markdown("**(विध AI) - Indian Legal Info Assistant**")
    st.markdown("---")
    st.markdown("⚠️ *This AI does not provide legal advice. It summarizes facts from Indian legal sources only.*")
    st.markdown("---")

st.title("🏛️ Ask Your Legal Question")
prompt = st.text_area("Your Question:", placeholder="e.g., What is the procedure for filing an FIR in India?", height=150)
submit_btn = st.button("Get Legal Info")

for key in ["prompt", "response_a", "response_b", "_clients"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key != "_clients" else None

if submit_btn:
    if not prompt:
        st.error("Please enter a question.", icon="✍️")
        st.stop()
    if not any(word in prompt.lower() for word in LEGAL_KEYWORDS):
        st.error("Please ask a question related to Indian law.", icon="⚖️")
        st.stop()

    if st.session_state["_clients"] is None:
        with st.spinner("🧠 Loading model & APIs..."):
            st.session_state["_clients"] = load_model()

    pipe, tavily, hf_api = st.session_state["_clients"]

    with st.spinner("⚖️ Searching trusted legal sources..."):
        context = get_search_context(prompt, tavily)

    if not context:
        st.error("Could not retrieve any context. Try again.", icon="🌐")
    else:
        with st.spinner("🧾 Generating responses..."):
            ans_a = generate_response(pipe, prompt, context, temperature=0.5)
            ans_b = generate_response(pipe, prompt, context, temperature=0.8)

        if ans_a and ans_b:
            st.session_state.prompt = prompt
            st.session_state.response_a = ans_a + "\n\n" + DISCLAIMER
            st.session_state.response_b = ans_b + "\n\n" + DISCLAIMER
        else:
            st.error("No valid response generated.", icon="🤖")

if st.session_state.response_a:
    st.subheader("Generated Responses")
    st.markdown("Please vote for the response that is more factual or helpful.")
    col1, col2 = st.columns(2)

    with col1:
        st.info("#### Response A")
        st.markdown(st.session_state.response_a)
        st.button("👍 Response A is better", on_click=handle_feedback_a, key="btn_a", use_container_width=True)

    with col2:
        st.info("#### Response B")
        st.markdown(st.session_state.response_b)
        st.button("👍 Response B is better", on_click=handle_feedback_b, key="btn_b", use_container_width=True)


