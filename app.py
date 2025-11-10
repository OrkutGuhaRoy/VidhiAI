# -------------------------------------------------------------------
#                   VidhiAI - app.py (Inference API Version)
# -------------------------------------------------------------------
# This app runs on your HF Space (CPU)
# It does NOT load the model. It CALLS the model's API.
# -------------------------------------------------------------------

import gradio as gr
import os
import json
import io
import uuid
from tavily import TavilyClient
from huggingface_hub import HfApi, InferenceClient

# -------------------------------------------------------------------
# 1. LOAD CONSTANTS AND SECRETS
# -------------------------------------------------------------------
print("Loading secrets and constants...")

# Load from HF Space Secrets
HF_TOKEN = os.getenv("HF_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# This is your Database Repo, loaded from secrets
# Set this secret to "SwastikGuhaRoy/VidhiAI-Response"
DB_REPO_ID = os.getenv("DB_REPO") 

# This gets the ID of the Space this code is running in.
# This will be "SwastikGuhaRoy/VidhiAI"
MODEL_REPO_ID = os.getenv("SPACE_ID") 

# -------------------------------------------------------------------
# 2. DEFINE ALL GUARDRAILS
# -------------------------------------------------------------------

# (Guardrails 1, 2, and 4 are the same)
LEGAL_KEYWORDS = ["law", "legal", "court", "act", "section", "ipc", "crpc", "fir", "india"]
TRUSTED_DOMAINS = [
    "indiacode.nic.in", "main.sci.gov.in", "prsindia.org",
    "www.livelaw.in", "www.barandbench.com"
]
DISCLAIMER = "\n\n**Disclaimer:** I am an AI assistant and not a lawyer. This information is for informational purposes only and does not constitute legal advice. Please consult a qualified legal professional for advice on your specific situation."

# Guardrail 3: System Prompt (This is now formatted for the API)
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

# -------------------------------------------------------------------
# 3. INITIALIZE API CLIENTS (Global)
# -------------------------------------------------------------------
# NO MODEL IS LOADED HERE!
print("Initializing API clients...")

# Initialize the API client to call our model
# This client will call the HF Serverless Inference API
client = InferenceClient(model=MODEL_REPO_ID, token=HF_TOKEN)

# Initialize other clients
tavily = TavilyClient(api_key=TAVILY_API_KEY)
hf_api = HfApi(token=HF_TOKEN)

# -------------------------------------------------------------------
# 4. HELPER FUNCTIONS (The Core Logic)
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
        return f"Error during search: {e}"

def generate_response(prompt, context, temperature=0.6):
    """
    Guardrail 3: Formats prompt and calls the SERVERLESS INFERENCE API.
    This is the new, lightweight function.
    """
    
    # Format the prompt using Llama 3's chat template
    # This is how the API expects it
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCONTEXT:\n{context}\n\nUSER QUESTION:\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    
    # This might take 20-60s on the first "cold start"
    # After that, it will be fast.
    try:
        response = client.text_generation(
            prompt=formatted_prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            stop_sequences=["<|eot_id|>"] # Stop when the model finishes
        )
        return response
    except Exception as e:
        print(f"Inference API Error: {e}")
        raise gr.Error(f"Model Inference Error: {e}. This can happen during a cold start. Please try again in 30 seconds.")


def save_feedback(prompt, chosen, rejected):
    """Saves feedback as a single JSON file to the HF Dataset repo."""
    # (This function is identical to the previous version)
    if not prompt or not chosen or not rejected:
        return "Feedback not saved (missing data)."
    if not HF_TOKEN or not DB_REPO_ID:
        return "Feedback not saved (server configuration error)."

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
        return "Feedback saved. Thank you!"
    except Exception as e:
        return f"Error saving feedback: {e}"

# -------------------------------------------------------------------
# 5. MAIN GRADIO FUNCTION (Identical to before)
# -------------------------------------------------------------------

def run_inference(prompt):
    """
    This function is triggered by the 'Submit' button.
    It runs all guardrails and returns two responses.
    """
    if not any(word in prompt.lower() for word in LEGAL_KEYWORDS):
        err_msg = "I am a legal assistant for India. Please ask a question about Indian law."
        return err_msg, err_msg
    
    context = get_search_context(prompt)
    if "Error" in context:
        return context, context
    
    # Generate two responses by calling the API
    answer_A = generate_response(prompt, context, temperature=0.5)
    answer_B = generate_response(prompt, context, temperature=0.8)
    
    final_A = answer_A + DISCLAIMER
    final_B = answer_B + DISCLAIMER
    
    return final_A, final_B

# -------------------------------------------------------------------
# 6. BUILD AND LAUNCH THE GRADIO APP (Line 169 is FIXED)
# -------------------------------------------------------------------
print("Building Gradio interface...")

#
#
# ----------------- THIS IS THE FIXED LINE -----------------
#
# We are using primary_hue="blue" which is a valid color palette
# Your hex code "#1E3A8A" is a shade of blue, so this will give a similar feel.
#
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as app:
#
#
# ----------------------------------------------------------
#
#
    gr.Markdown(
        """
        # 🏛️ VidhiAI - Indian Legal Info Assistant
        **This is a test environment.** Responses are generated by an AI.
        (First query may take up to 60s to "wake up" the model).
        """
    )
    
    with gr.Column():
        prompt_input = gr.Textbox(label="Ask your legal question", placeholder="e.g., What is the procedure for filing an FIR in India?")
        submit_btn = gr.Button("Submit", variant="primary")
        feedback_status = gr.Textbox(label="Feedback Status", interactive=False, placeholder="Vote for a response to save feedback...")
        
    with gr.Row():
        response_A = gr.Textbox(label="Response A", lines=15, interactive=False)
        response_B = gr.Textbox(label="Response B", lines=15, interactive=False)
        
    with gr.Row():
        choose_A = gr.Button("👍 Response A is better")
        choose_B = gr.Button("👍 Response B is better")

    submit_btn.click(
        fn=run_inference,
        inputs=[prompt_input],
        outputs=[response_A, response_B]
    )
    choose_A.click(
        fn=save_feedback,
        inputs=[prompt_input, response_A, response_B],
        outputs=[feedback_status]
    )
    choose_B.click(
        fn=save_feedback,
        inputs=[prompt_input, response_B, response_A],
        outputs=[feedback_status]
    )

print("Gradio app built. Launching...")
if __name__ == "__main__":
    app.launch()