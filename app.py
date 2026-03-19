# app.py
import os
import io
import json
import re
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("Attempted to load environment variables from .env")

app = Flask(__name__)

# --- Configuration ---
# No explicit upload file size limit.

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

# --- Background job state ---
jobs = {}  # job_id -> job info dict
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", "2")))


def _get_job(job_id):
    with jobs_lock:
        return jobs.get(job_id)


def _update_job(job_id, **fields):
    with jobs_lock:
        if job_id not in jobs:
            return
        jobs[job_id].update(fields)


def process_classification_job(
    job_id,
    file_bytes,
    filename,
    text_column_name,
    categories_list,
    provider_name,
    model_name,
    api_key,
    api_base,
):
    """
    Run classification in a background thread and store the CSV + progress in the global `jobs` dict.
    """
    import pandas as pd

    cancel_event = None
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        cancel_event = job.get("cancel_event")

    def is_cancelled():
        return cancel_event is not None and cancel_event.is_set()

    _update_job(job_id, status="running", progress_current=0)
    try:
        file_stream = io.BytesIO(file_bytes)
        df = None
        if filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(file_stream, encoding="utf-8-sig")
            except UnicodeDecodeError:
                file_stream.seek(0)
                df = pd.read_csv(file_stream, encoding="utf-8")
        elif filename.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_stream, engine="openpyxl")
        else:
            raise ValueError("Unsupported file type. Use .csv, .xls, .xlsx.")

        total_rows = len(df)
        _update_job(job_id, progress_total=total_rows, message=f"Found {total_rows} rows.")

        results_labels = []
        results_justifications = []
        row_count = 0

        if total_rows == 0:
            df["Assigned Category"] = []
            df["Justification"] = []
        else:
            print(f"[job {job_id}] Starting {provider_name} classification with {model_name} for {total_rows} rows...")
            for _, row in df.iterrows():
                if is_cancelled():
                    _update_job(job_id, status="cancelled", message="Cancelled by user.")
                    return

                row_count += 1
                text_to_classify_raw = row[text_column_name]
                text_to_classify = str(text_to_classify_raw) if pd.notna(text_to_classify_raw) else ""
                if not text_to_classify.strip():
                    label, justification = "no category", "Text field empty."
                else:
                    llm_result = classify_text_with_llm(
                        text_to_classify,
                        categories_list,
                        provider_name,
                        model_name,
                        api_key,
                        api_base,
                    )
                    label = llm_result.get("label", "error")
                    justification = llm_result.get("justification", "Error retrieving justification.")

                results_labels.append(label)
                results_justifications.append(justification)

                # Update progress occasionally to reduce contention.
                if row_count % 5 == 0 or row_count == total_rows:
                    _update_job(
                        job_id,
                        progress_current=row_count,
                        message=f"Processed {row_count}/{total_rows} rows...",
                    )

            df["Assigned Category"] = results_labels
            df["Justification"] = results_justifications

        # Generate output CSV in memory
        output_csv_stream = io.StringIO()
        df.to_csv(output_csv_stream, index=False, encoding="utf-8")
        csv_data = output_csv_stream.getvalue()
        output_csv_stream.close()

        _update_job(job_id, status="done", progress_current=row_count, csv_data=csv_data, message="Done.")
    except Exception as e:
        _update_job(job_id, status="error", message=str(e))
        print(f"[job {job_id}] ERROR: {e}")



# --- Custom Error Handler for 413 Request Entity Too Large ---
@app.errorhandler(413)
def request_entity_too_large(error):
    """ Custom JSON response for file size limit exceeded. """
    print("Error 413: Request entity too large.")
    return jsonify(error="File size exceeds the server limit. Please upload a smaller file."), 413

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
    # Lazy import to keep app startup fast for Render health checks.
    from litellm import completion

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
    print("\n=== Received request on /classify (job start) ===")

    # 1. --- Input Validation ---
    if 'csv_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['csv_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

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

    if not request.form.get('text_column'):
        return jsonify({"error": "Text column name missing"}), 400
    if not request.form.get('categories'):
        return jsonify({"error": "Categories definition missing"}), 400

    text_column_name = request.form['text_column']
    categories_json_string = request.form['categories']

    try:
        categories_list = json.loads(categories_json_string)
        if not isinstance(categories_list, list) or not categories_list:
            raise ValueError("Categories must be a non-empty list.")
        for item in categories_list:
            if not isinstance(item, dict) or 'label' not in item or 'description' not in item:
                raise ValueError("Invalid category structure.")
            if not item['label'].strip() or not item['description'].strip():
                raise ValueError("Category label/desc empty.")
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Invalid categories format: {e}"}), 400

    filename = file.filename
    file_bytes = file.stream.read()
    job_id = str(uuid.uuid4())
    cancel_event = threading.Event()

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress_current": 0,
            "progress_total": 0,
            "message": "Queued.",
            "csv_data": None,
            "error": None,
            "cancel_event": cancel_event,
        }

    executor.submit(
        process_classification_job,
        job_id,
        file_bytes,
        filename,
        text_column_name,
        categories_list,
        provider_name,
        model_name,
        api_key,
        api_base,
    )

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route('/classify/progress', methods=['GET'])
@limiter.exempt
def classify_progress():
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(
        {
            "job_id": job_id,
            "status": job.get("status"),
            "progress_current": job.get("progress_current", 0),
            "progress_total": job.get("progress_total", 0),
            "message": job.get("message", ""),
        }
    )


@app.route('/classify/cancel', methods=['POST'])
@limiter.exempt
def classify_cancel():
    job_id = request.args.get("job_id", "").strip() or request.form.get("job_id", "").strip()
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    current_status = job.get("status")
    if current_status in ("done", "error", "cancelled"):
        return jsonify({"job_id": job_id, "status": current_status})
    cancel_event = job.get("cancel_event")
    if cancel_event:
        cancel_event.set()
    _update_job(job_id, status="cancelling", message="Cancelling...")
    return jsonify({"job_id": job_id, "status": "cancelling"})


@app.route('/classify/result', methods=['GET'])
@limiter.exempt
def classify_result():
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    status = job.get("status")
    if status in ("queued", "running", "cancelling"):
        return jsonify({"error": f"Job not ready. Current status: {status}"}), 425
    if status == "done":
        csv_data = job.get("csv_data") or ""
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment;filename=classified_output.csv",
                "Content-Type": "text/csv; charset=utf-8",
            },
        )
    if status == "cancelled":
        return jsonify({"error": "Job cancelled"}), 410
    if status == "error":
        return jsonify({"error": job.get("message", "Job failed")}), 500
    return jsonify({"error": f"Unhandled job status: {status}"}), 500

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

