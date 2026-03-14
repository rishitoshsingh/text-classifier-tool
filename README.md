# Text Classification Tool

## Beta Version available at (limited to file <= 30KB):

https://classifyit.netlify.app/

## Description

A web application that allows users to upload a CSV/Excel file, define custom labels along with their descriptions, then download a classified version of the file after submission.

## Features

* Upload CSV or Excel files.
* Define custom classification labels (up to 50 chars: letters, numbers, spaces, underscore) and descriptions (up to 300 chars).
* Limit of 10 categories per classification task.
* Save and load category definitions locally in the browser (`localStorage`).
* Specify the column containing the text to classify (up to 50 chars: letters, numbers, spaces, underscore).
* Backend processing using LiteLLM with selectable providers such as OpenAI, Gemini, Groq, Anthropic, and OpenAI-compatible endpoints.
* Frontend controls for choosing the model provider, selecting a model, and entering the API key to use for the current request.
* Support for OpenAI-compatible APIs with a custom `base_url` and custom model name.
* Provides justification from the LLM for each classification.
* Handles missing/empty text fields.
* Returns a downloadable CSV file with original text, assigned category, and justification.
* Real-time input validation for category definitions and column name.
* Backend validation for column name existence in the uploaded file.
* Option to clear the selected file.
* Single-service deployment: Flask now serves the frontend too, so you can deploy the whole app as one web service.

## Recommended Deployment

For the easiest handoff, deploy this as a single Flask service.

Best choice: `Render`
* Free web service tier is available.
* The frontend and backend are deployed together.
* This repo already includes `render.yaml`, `Procfile`, `Dockerfile`, and `.env.example`.
* Someone else can deploy it with almost no code changes.

Free-tier caveat:
* Render free services sleep after inactivity, so the first request can be slow.

## Free Options

### 1. Render (Recommended)
Best for: easiest deployment by another person.

Why:
* Free tier for small hobby apps.
* Python support is straightforward.
* One service deploy is enough for this repo.
* `render.yaml` is already included.

### 2. Cloudflare Pages + separate backend
Best for: free static hosting only.

Why:
* Excellent free static hosting.
* Good if you later split frontend and backend again.

Downside:
* Not the easiest handoff, because someone must also deploy the Flask backend somewhere else.

### 3. Railway
Best for: easy deploy experience, but not truly free long term.

Why:
* Very simple deployment flow.

Downside:
* Mostly trial / credit-based now, so I would not treat it as a dependable free option.

## Setup (Local Development)

1.  **Clone the repository:**
    ```bash
    # Replace with your actual repository URL after creating it on GitHub
    git clone [https://github.com/darLloyd/your-repo-name.git](https://github.com/darLloyd/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Ensure Python 3.7+ is installed
    python -m venv venv
    # Activate the environment
    source venv/bin/activate  # macOS/Linux
    # OR
    venv\Scripts\activate  # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `requirements.txt` includes Flask, pandas, litellm, python-dotenv, Flask-CORS, openpyxl)*
4.  **Create `.env` file (optional):** Copy `.env.example` to `.env` if you want server-side fallback keys for local development:
    ```plaintext
    cp .env.example .env
    ```
    Then fill any provider values you want:
    ```plaintext
    OPENAI_API_KEY=sk-YourActualOpenAIKeyHere...
    GEMINI_API_KEY=your-gemini-key
    GROQ_API_KEY=your-groq-key
    ANTHROPIC_API_KEY=your-anthropic-key
    OPENAI_COMPATIBLE_API_KEY=your-compatible-key
    OPENAI_COMPATIBLE_BASE_URL=https://your-endpoint.example.com/v1
    ```
    You can also leave these unset and enter the provider API key directly in the web app each time.
5.  **Run the backend server:**
    ```bash
    # Set environment variable for Flask (do this once per terminal session or add to system variables)
    export FLASK_APP=app.py # macOS/Linux
    # OR set FLASK_APP=app.py # Windows CMD
    # OR $env:FLASK_APP = "app.py" # Windows PowerShell

    # Run the development server
    flask run
    ```
    The backend should now be running, typically on `http://127.0.0.1:5000`. Check the terminal output.

6.  **Open the app:** Visit `http://127.0.0.1:5000` in your browser.

## Fastest Local Start

If someone just wants to run it locally:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
flask --app app.py run
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Docker Start

If someone prefers Docker:

```bash
docker build -t text-classifier-tool .
docker run --env-file .env -p 5000:5000 text-classifier-tool
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Render Deploy

This repo is already prepared for Render.

### Option A: easiest for another person
1. Fork or clone the repo to GitHub.
2. Create a new Render account.
3. In Render, choose `New +` -> `Blueprint`.
4. Select the repo.
5. Render will detect `render.yaml`.
6. Add any environment variables if desired, or leave them blank and let end users enter API keys in the UI.
7. Deploy.

### Option B: regular web service
If they do not want to use Blueprint:
1. Create a new `Web Service` in Render.
2. Connect the repo.
3. Use:
   * Build command: `pip install -r requirements.txt`
   * Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1`
4. Deploy on the free plan.

Render note:
* Classification can take longer than 30 seconds because the app processes rows one by one and makes LLM calls.
* A default Gunicorn timeout can cause `502` errors on Render for longer jobs.
* This repo now uses a higher timeout in `render.yaml`, `Procfile`, and `Dockerfile`.

## Usage

1.  Open the deployed app URL in your browser.
2.  Choose a model provider.
3.  Select a model from the dropdown. If you choose `OpenAI-Compatible`, enter a custom model name and base URL instead.
4.  Enter the API key you want to use for this run.
5.  Click "Choose File" and select a CSV or Excel file. If you select the wrong file, click the "×" button to clear it.
6.  Enter the exact name of the column in your file that contains the text you want to classify (validation rules apply).
7.  Define at least one category by entering a Label and Description (validation rules apply). Use the "+ Add Category" button for more (up to 10).
    * *(Optional)* Use "Save Categories" to store definitions in your browser for later use.
    * *(Optional)* Use "Load Categories" to load previously saved definitions.
8.  Click "Submit for Classification".
9.  Wait for processing. Monitor the backend terminal for progress logs and potential errors from the selected provider or file processing.
10.  If successful, a classified CSV file (`classified_output_....csv`) will be downloaded automatically. Check the "Status" area in the web app for messages or errors reported back from the backend.

## Future Enhancements

* Add one-click deploy buttons for Render and similar hosts.
* Add GitHub Actions for automatic validation before deploy.
* Implement backend rate limiting and row limits for hosted version.
* Add option to specify Excel sheet name if multiple sheets exist.
* More robust error handling and user feedback (e.g., specific parsing errors).
* Add unit/integration tests.
* (Add other ideas here)

