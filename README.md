# LexiGuard-AI
GenAI Hackathon (Phase 2) 🚀 | An intelligent legal co-pilot built with Gemini 1.5 Pro, Document AI, and FastAPI. Deployed on Google Cloud Run.

**A GenAI Hackathon Project**

This project is a web-based legal analysis tool designed to "demystify" complex legal documents. It leverages Google's Gemini 1.5 Pro and Document AI to provide users with a clear, actionable analysis of their contracts, agreements, and other legal texts.

The application supports multiple jurisdictions, cross-referencing document text against a global database of legal principles to identify risks, provide recommendations, and even generate a "Legal Compliance Score."

![Project Screenshot](image_5fa586.png)

---

## Problem Statement

Legal documents are dense, full of jargon, and highly specific to jurisdictional laws. For small businesses or individuals, hiring a lawyer to review every document (like a rental agreement, an employment contract, or a freelance agreement) is time-consuming and expensive. This creates a barrier to understanding one's own legal rights and risks.

## Our Solution

The **Global Legal AI Demystifier** is a "legal co-pilot" that solves this problem by providing an instant, AI-powered first-pass analysis.

A user can upload any legal document (PDF, DOCX, or even a photo) and select their country. The application will:
1.  [cite_start]**Extract Text:** Use **Google Document AI** for high-fidelity OCR to digitize the text[cite: 367].
2.  [cite_start]**Analyze Risk:** Use **Gemini 1.5 Pro** to read the text, identify key clauses, and check for risks, unenforceable clauses, or vague language [cite: 316-356].
3.  **Provide Context:** Cross-reference the document's terms with an internal **global law database** (SQLite) containing legal principles for various countries.
4.  **Deliver a Report:** Present the user with a simple, user-friendly dashboard that includes:
    * An Executive Summary
    * A "High Risk" and "Moderate Risk" analysis
    * Specific Legal Recommendations
    * A "Legal Compliance Score"

---

## 🌟 Key Features

* [cite_start]**📄 Multi-Format Document Upload:** Accepts `.pdf`, `.docx`, `.txt`, and images (`.jpg`, `.png`) [cite: 400-403].
* **🤖 Advanced AI Analysis:** Uses **Gemini 1.5 Pro** for nuanced legal reasoning and risk identification.
* [cite_start]**🔍 High-Fidelity OCR:** Leverages **Google Document AI** to extract text from scanned documents and images[cite: 367].
* [cite_start]**🌍 Global Jurisdiction Support:** Users can select their country (e.g., India, USA, UK, Germany, Japan) for a jurisdiction-specific analysis [cite: 106-121].
* **⚡ Async Processing:** File uploads are instant. Slow OCR and AI analysis are handled by **FastAPI Background Tasks** for a non-blocking user experience.
* [cite_start]**💬 Interactive Q&A:** A "Legal Consultation" chatbot to ask follow-up questions about the analyzed document[cite: 638].
* **🤝 Negotiation Simulator:** A chat interface to practice negotiating difficult clauses.
* **📥 Downloadable PDF Reports:** Generate and download a professional PDF summary of the AI's findings.

---

## 🛠️ Tech Stack

* **Backend:** Python 3.11, FastAPI, Uvicorn
* **AI Services:** Google Generative AI (Gemini 1.5 Pro), Google Document AI
* **Database:** SQLite (for the global law database), SQLAlchemy (ORM)
* **File Processing:** PyMuPDF (fitz), python-docx
* **Frontend:** Vanilla HTML, CSS, and JavaScript (served directly from FastAPI)
* **Deployment:** Docker, Google Cloud Run

---

## 🏗️ Application Architecture



1.  **User** uploads a file via the HTML/JS frontend.
2.  **FastAPI** backend receives the file. It instantly saves the file to a temporary location (`/tmp`) and creates a "PENDING" job in the SQLite database.
3.  The frontend polls the `/analyze/status/{job_id}` endpoint.
4.  A **Background Task** picks up the job, reads the file, and sends it to **Google Document AI** for OCR.
5.  The extracted text is sent to the **Gemini 1.5 Pro API** along with context from the law database.
6.  The AI's JSON response is saved to the job in the database, and the status is set to "COMPLETE".
7.  The frontend receives the "COMPLETE" status, fetches the JSON results, and renders the dashboard.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11+
* A Google Cloud Project
* **Google Cloud SDK** (`gcloud`) installed and authenticated locally:
    ```bash
    gcloud auth application-default login
    ```
* Enabled APIs on your Google Cloud project:
    * Google Generative AI (Vertex AI or AI Platform)
    * Document AI API

### 1. Configuration

Create a `.env` file in the root directory and add your Google API key:

```.env
# Get this from Google AI Studio or your GCP project
GOOGLE_API_KEY="AIzaSy...your...key...here"

# These are hardcoded in allfinal.py but can be overridden
# PROJECT_ID="631766745811"
# LOCATION="us"
# PROCESSOR_ID="6676c1eecd31b45"
```

**Note:** The Document AI processor details are currently hardcoded in `allfinal.py` but can be moved to the `.env` file.

### 2. Local Setup & Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sanjay-Program/LexiGuard-AI/
    cd legal-ai
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows: myenv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the application:**
    ```bash
    python allfinal.py
    ```
    The app will be running at `http://127.0.0.1:8000`.

### 3. Deployment to Google Cloud Run

This application is designed for a one-click deployment to Google Cloud Run.

1.  **Set your Project ID:**
    ```bash
    gcloud config set project [YOUR_PROJECT_ID]
    ```
2.  **Enable Required APIs:**
    ```bash
    gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com documentai.googleapis.com secretmanager.googleapis.com
    ```
3.  **Create an Artifact Registry repository:**
    ```bash
    gcloud artifacts repositories create legal-ai-repo --repository-format=docker --location=asia-south1
    ```
4.  **Create a Service Account for Cloud Run:**
    ```bash
    gcloud iam service-accounts create legal-ai-sa --display-name="Legal AI Cloud Run SA"
    ```
5.  **Give it Document AI permissions:**
    ```bash
    gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] --member="serviceAccount:legal-ai-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com" --role="roles/documentai.apiUser"
    ```
6.  **Store your Gemini API Key in Secret Manager:**
    ```bash
    echo -n "AIzaSy...your...key...here" | gcloud secrets create GOOGLE_API_KEY --data-file=-
    ```
7.  **Give the Service Account access to the secret:**
    ```bash
    gcloud secrets add-iam-policy-binding GOOGLE_API_KEY --member="serviceAccount:legal-ai-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
    ```
8.  **Build the container using Cloud Build:**
    ```bash
    gcloud builds submit . --tag asia-south1-docker.pkg.dev/[YOUR_PROJECT_ID]/legal-ai-repo/legal-ai-service:latest
    ```
9.  **Deploy to Cloud Run:**
    ```bash
    gcloud run deploy legal-ai-service \
        --image=asia-south1-docker.pkg.dev/[YOUR_PROJECT_ID]/legal-ai-repo/legal-ai-service:latest \
        --region=asia-south1 \
        --platform=managed \
        --service-account=legal-ai-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com \
        --set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest \
        --port=8000 \
        --allow-unauthenticated
    ```
Your service will be live at the URL provided by the deploy command.

---

## 👨‍💻 Author

* **Visoneers**
* Contact: `sanjaymurugadoss02@gmail.com`
