# app.py
import os
import io
import json
import re
import pandas as pd
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from litellm import completion

# Load environment variables
load_dotenv()
print("Attempted to load environment variables from .env")

app = Flask(__name__)

# --- Configuration ---
# ** UPDATED File Size Limit (30 KB = 30 * 1024 bytes) **
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024
print(f"MAX_CONTENT_LENGTH set to {app.config['MAX_CONTENT_LENGTH']} bytes ({app.config['MAX_CONTENT_LENGTH']/1024:.0f} KB)")

# CORS Configuration
CORS(app)
print("Flask-CORS initialized.")

# --- Rate Limiter Configuration --- (Same as before)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "20 per hour"],
    storage_uri="memory://",
)
print("Flask-Limiter initialized.")

# --- LiteLLM Provider Configuration ---
PROVIDER_CONFIGS = {
    "openai": {
        "display_name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "default_model": "openai/gpt-4o-mini",
        "models": {
            "openai/gpt-4o-mini": "GPT-4o mini",
            "openai/gpt-4.1-mini": "GPT-4.1 mini",
            "openai/gpt-4.1-nano": "GPT-4.1 nano",
        },
    },
    "gemini": {
        "display_name": "Gemini",
        "env_var": "GEMINI_API_KEY",
        "default_model": "gemini/gemini-2.0-flash-lite",
        "models": {
            "gemini/gemini-2.0-flash-lite": "Gemini 2.0 Flash-Lite",
            "gemini/gemini-2.0-flash": "Gemini 2.0 Flash",
            "gemini/gemini-1.5-flash": "Gemini 1.5 Flash",
        },
    },
    "groq": {
        "display_name": "Groq",
        "env_var": "GROQ_API_KEY",
        "default_model": "groq/llama-3.1-8b-instant",
        "models": {
            "groq/llama-3.1-8b-instant": "Llama 3.1 8B Instant",
            "groq/gemma2-9b-it": "Gemma 2 9B IT",
            "groq/llama-3.3-70b-versatile": "Llama 3.3 70B Versatile",
        },
    },
    "anthropic": {
        "display_name": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "anthropic/claude-3-5-haiku-latest",
        "models": {
            "anthropic/claude-3-5-haiku-latest": "Claude 3.5 Haiku",
        },
    },
    "openai_compatible": {
        "display_name": "OpenAI-Compatible",
        "env_var": "OPENAI_COMPATIBLE_API_KEY",
        "env_base_var": "OPENAI_COMPATIBLE_BASE_URL",
        "allow_custom_model": True,
        "requires_base_url": True,
    },
}

configured_providers = [
    config["display_name"]
    for config in PROVIDER_CONFIGS.values()
    if os.getenv(config["env_var"])
]
if configured_providers:
    print(f"LiteLLM providers configured via environment: {', '.join(configured_providers)}")
else:
    print("\n*** WARNING: No provider API keys configured in environment. Users must supply an API key in the UI. ***\n")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# --- Custom Error Handler for 413 Request Entity Too Large ---
@app.errorhandler(413)
def request_entity_too_large(error):
    """ Custom JSON response for file size limit exceeded. """
    # ** UPDATED error message to reflect 30KB limit **
    limit_kb = app.config['MAX_CONTENT_LENGTH']/1024
    print(f"Error 413: Request entity too large (limit: {limit_kb:.0f} KB).")
    return jsonify(error=f"File size exceeds limit ({limit_kb:.0f} KB). Please upload a smaller file."), 413

# --- Custom Error Handler for 429 Rate Limit Exceeded --- (Same as before)
@app.errorhandler(429)
def ratelimit_handler(e):
    """ Custom JSON response for rate limit exceeded. """
    print(f"Error 429: Rate limit exceeded. Description: {e.description}")
    return jsonify(error=f"Rate limit exceeded: {e.description}"), 429


# --- LLM Classification Function ---
def get_provider_config(provider_name):
    return PROVIDER_CONFIGS.get((provider_name or "").strip().lower())


def get_api_key(provider_name, request_api_key=""):
    provider_config = get_provider_config(provider_name)
    if not provider_config:
        return ""
    if request_api_key and request_api_key.strip():
        return request_api_key.strip()
    return os.getenv(provider_config["env_var"], "").strip()


def get_api_base(provider_name, request_api_base=""):
    provider_config = get_provider_config(provider_name)
    if not provider_config or not provider_config.get("requires_base_url"):
        return ""
    if request_api_base and request_api_base.strip():
        return request_api_base.strip().rstrip("/")
    env_base_var = provider_config.get("env_base_var")
    return os.getenv(env_base_var, "").strip().rstrip("/") if env_base_var else ""


def get_model_name(provider_name, requested_model_name=""):
    provider_config = get_provider_config(provider_name)
    if not provider_config:
        return None
    model_name = (requested_model_name or "").strip()
    if provider_config.get("allow_custom_model"):
        if not model_name:
            return None
        return model_name if model_name.startswith("openai/") else f"openai/{model_name}"
    if not model_name:
        return provider_config["default_model"]
    if model_name in provider_config["models"]:
        return model_name
    return None


def classify_text_with_llm(text_to_classify, categories, provider_name, model_name, api_key, api_base=""):
    provider_config = get_provider_config(provider_name)
    if not provider_config:
        return {"label": "error", "justification": "Unsupported model provider."}
    if not model_name:
        return {"label": "error", "justification": "Unsupported model for selected provider."}
    if not api_key:
        return {
            "label": "error",
            "justification": f"{provider_config['display_name']} API key not provided.",
        }
    if not text_to_classify:
        return {"label": "no category", "justification": "Input text was empty."}

    category_definitions = "\n".join([f"- **{cat['label']}**: {cat['description']}" for cat in categories])
    system_prompt = f"""
You are a text classification assistant. Your task is to classify the user's text based *only* on the categories provided below.
**Available Categories:**
{category_definitions}
**Instructions:** 1. Read text and descriptions. 2. Choose best matching label or "no category". 3. Provide brief justification.
**Output Format:** Respond *only* with:
Label: <Chosen Label or "no category">
Justification: <Your brief explanation>
"""
    user_prompt = f"Please classify the following text:\n\n\"{text_to_classify}\""
    print(
        f"\n--- Sending prompt to {provider_config['display_name']} ({model_name}) "
        f"for text starting: '{text_to_classify[:60]}...' ---"
    )
    try:
        completion_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 100,
            "api_key": api_key,
        }
        if api_base:
            completion_kwargs["api_base"] = api_base
        response = completion(**completion_kwargs)
        print(f"--- Received {provider_config['display_name']} response ---")
        response_content = response.choices[0].message.content
        response_text = response_content.strip() if isinstance(response_content, str) else str(response_content).strip()
        label, justification = "error", "Could not parse LLM response."
        label_match = re.search(r"^Label:\s*(.*)", response_text, re.MULTILINE | re.IGNORECASE)
        justification_match = re.search(r"^Justification:\s*(.*)", response_text, re.MULTILINE | re.IGNORECASE)
        if label_match:
            label = label_match.group(1).strip()
            valid_labels = [cat['label'] for cat in categories] + ['no category']
            if label.lower() not in [vl.lower() for vl in valid_labels]:
                print(f"Warning: {provider_config['display_name']} returned invalid label '{label}'.")
        if justification_match: justification = justification_match.group(1).strip()
        if label == "error" and response_text: justification = f"Failed parse. Raw: {response_text[:100]}..."
        print(f"Parsed Label: '{label}', Justification: '{justification[:60]}...'")
        return {"label": label, "justification": justification}
    except Exception as e:
        print(f"!!! ERROR during {provider_config['display_name']} call/parsing: {e}")
        justification = f"{provider_config['display_name']} API call failed: {e}"
    return {"label": "error", "justification": justification}


# --- API Endpoint for Classification ---
@app.route('/classify', methods=['POST'])
@limiter.exempt
def classify_csv():
    print("\n=== Received request on /classify ===")
    # 1. --- Input Validation ---
    if 'csv_file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['csv_file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    provider_name = request.form.get('model_provider', '').strip().lower()
    provider_config = get_provider_config(provider_name)
    if not provider_config:
        supported_providers = ", ".join(PROVIDER_CONFIGS.keys())
        return jsonify({"error": f"Unsupported model provider. Choose one of: {supported_providers}"}), 400
    model_name = get_model_name(provider_name, request.form.get('model_name', ''))
    if not model_name:
        return jsonify({"error": f"Unsupported model for {provider_config['display_name']}."}), 400
    api_base = get_api_base(provider_name, request.form.get('base_url', ''))
    if provider_config.get("requires_base_url") and not api_base:
        return jsonify({"error": f"Base URL missing for {provider_config['display_name']}."}), 400
    api_key = get_api_key(provider_name, request.form.get('api_key', ''))
    if not api_key:
        return jsonify({"error": f"API key missing for {provider_config['display_name']}."}), 400
    if not request.form.get('text_column'): return jsonify({"error": "Text column name missing"}), 400
    if not request.form.get('categories'): return jsonify({"error": "Categories definition missing"}), 400
    text_column_name = request.form['text_column']
    categories_json_string = request.form['categories']
    print(f"Using provider: {provider_config['display_name']} ({model_name})")
    if api_base:
        print(f"Using API base: {api_base}")
    try: # Category validation
        categories_list = json.loads(categories_json_string)
        if not isinstance(categories_list, list) or not categories_list: raise ValueError("Categories must be a non-empty list.")
        for item in categories_list:
            if not isinstance(item, dict) or 'label' not in item or 'description' not in item: raise ValueError("Invalid category structure.")
            if not item['label'].strip() or not item['description'].strip(): raise ValueError("Category label/desc empty.")
        print(f"Received {len(categories_list)} valid categories.")
    except (json.JSONDecodeError, ValueError) as e: return jsonify({"error": f"Invalid categories format: {e}"}), 400

    # 2. --- File Parsing (CSV or Excel) ---
    filename = file.filename; df = None; print(f"Parsing file: {filename}")
    try:
        if filename.lower().endswith('.csv'):
            print("Reading CSV...")
            file_stream = io.BytesIO(file.stream.read())
            try: df = pd.read_csv(file_stream, encoding='utf-8-sig')
            except UnicodeDecodeError: file_stream.seek(0); df = pd.read_csv(file_stream, encoding='utf-8')
            except Exception as csv_e: raise ValueError(f"CSV parsing error: {csv_e}") from csv_e
        elif filename.lower().endswith(('.xls', '.xlsx')):
            print(f"Reading Excel ({filename.split('.')[-1]})...")
            file_stream = io.BytesIO(file.stream.read())
            df = pd.read_excel(file_stream, engine='openpyxl')
        else: return jsonify({"error": f"Unsupported file type. Use .csv, .xls, .xlsx."}), 400
        print(f"Parsed DataFrame shape: {df.shape}")
        if df.empty and len(df.columns) == 0: raise pd.errors.EmptyDataError("File empty.")
        if text_column_name not in df.columns: available_columns = list(df.columns); return jsonify({"error": f"Column '{text_column_name}' not found.", "available_columns": available_columns}), 400
    except UnicodeDecodeError as e: return jsonify({"error": f"File encoding error: {e}. Use UTF-8."}), 400
    except pd.errors.EmptyDataError as e: return jsonify({"error": f"File contains no data: {e}"}), 400
    except ImportError: return jsonify({"error": "Need 'openpyxl' for Excel. `pip install openpyxl`."}), 500
    except Exception as e: print(f"Error parsing file: {e}"); return jsonify({"error": f"Error reading file: {e}"}), 400

    # 3. --- Processing Loop & LLM Calls ---
    results_labels = []; results_justifications = []; row_count = 0
    if df.empty: print("DataFrame empty. Skipping processing.")
    else:
        print(f"Starting {provider_config['display_name']} classification with {model_name} for {len(df)} rows...")
        try:
            for index, row in df.iterrows():
                 row_count += 1; text_to_classify_raw = row[text_column_name]; text_to_classify = str(text_to_classify_raw) if pd.notna(text_to_classify_raw) else ""
                 if not text_to_classify.strip(): label, justification = "no category", "Text field empty."
                 else: llm_result = classify_text_with_llm(text_to_classify, categories_list, provider_name, model_name, api_key, api_base); label = llm_result.get("label", "error"); justification = llm_result.get("justification", "Error retrieving justification.")
                 results_labels.append(label); results_justifications.append(justification)
                 if row_count % 10 == 0 or row_count == len(df): print(f"  Processed {row_count}/{len(df)} rows...")
        except Exception as e: print(f"!!! Error during processing loop near row {row_count}: {e}"); return jsonify({"error": f"Processing error near row {row_count}: {e}"}), 500
        print(f"Finished {provider_config['display_name']} classification processing.")
    df['Assigned Category'] = results_labels; df['Justification'] = results_justifications

    # 4. --- Generate Output CSV ---
    try:
        output_csv_stream = io.StringIO();
        if 'Assigned Category' not in df.columns: df['Assigned Category'] = []
        if 'Justification' not in df.columns: df['Justification'] = []
        df.to_csv(output_csv_stream, index=False, encoding='utf-8'); csv_data = output_csv_stream.getvalue(); output_csv_stream.close(); print("Generated output CSV.")
    except Exception as e: print(f"Error generating output CSV: {e}"); return jsonify({"error": f"Failed to generate output CSV: {e}"}), 500

    # 5. --- Return CSV Response ---
    print("Sending CSV response.")
    return Response( csv_data, mimetype="text/csv", headers={ "Content-Disposition": "attachment;filename=classified_output.csv", "Content-Type": "text/csv; charset=utf-8" } )

@app.route('/')
def index():
    return send_from_directory(PROJECT_ROOT, 'index.html')


@app.route('/privacy')
@app.route('/privacy.html')
def privacy():
    return send_from_directory(PROJECT_ROOT, 'privacy.html')


@app.route('/health')
@limiter.exempt
def health():
    return jsonify({"status": "ok"})

# Main execution block (Same)
if __name__ == '__main__':
    if not configured_providers:
        print("\n--- NOTE: Server starting without provider API keys in environment. Users must supply one in the UI. ---\n")
    app.run(debug=True, port=int(os.getenv("PORT", "5000")))

