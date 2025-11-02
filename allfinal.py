# legal_ai_global_complete.py
# Complete Legal AI Demystifier with Global Laws Database & Country Selection
# Run: python legal_ai_global_complete.py

import os
import io
import re
import asyncio
import json
import uuid
import base64
import glob
from typing import List, Optional, Dict, Any
from html import unescape
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

# Core dependencies
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# AI & ML
import google.generativeai as genai

# Document AI OCR
from google.cloud import documentai

# File processing
import docx
import fitz  # PyMuPDF

# PDF export
from fpdf import FPDF

# Database (SQLAlchemy)
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func, Enum as SQLEnum
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# --- CONFIGURATION ---
load_dotenv()

class AnalysisStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING" 
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")

# Your Document AI processor details
PROJECT_ID = "631766745811"
LOCATION = "us"
PROCESSOR_ID = "6676c1eecd31b45"
DOC_AI_PROCESSOR_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

print(f"Using Document AI Processor: {DOC_AI_PROCESSOR_NAME}")

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-pro')
    print("✅ Successfully configured Google AI.")
except Exception as e:
    print(f"⚠️ Failed to configure Google AI: {e}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/docai-accessor-key.json"

# --- DATABASE SETUP ---
DB_PATH = os.getenv("LAW_DB_PATH", "legal_ai_global.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Law(Base):
    __tablename__ = "laws"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(400), nullable=False)
    jurisdiction = Column(String(120), default="India")
    country = Column(String(100), default="India")
    tags = Column(String(400), default="")
    text = Column(Text, nullable=False)
    category = Column(String(200), default="General")
    source_file = Column(String(300), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, index=True, nullable=False)
    status = Column(SQLEnum(AnalysisStatus), default=AnalysisStatus.PENDING)
    file_path = Column(String(1024), nullable=True)
    user_name = Column(String(200), nullable=True)
    language = Column(String(10), default="en")
    country = Column(String(100), default="India")
    analysis_result_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Drop and recreate tables to ensure schema matches
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- GLOBAL COUNTRY & LANGUAGE SUPPORT ---
COUNTRIES = {
    "India": {
        "languages": ["en", "hi", "ta", "te", "ml", "kn"],
        "legal_system": "Common Law",
        "currency": "INR"
    },
    "USA": {
        "languages": ["en"],
        "legal_system": "Common Law", 
        "currency": "USD"
    },
    "UK": {
        "languages": ["en"],
        "legal_system": "Common Law",
        "currency": "GBP"
    },
    "Japan": {
        "languages": ["ja", "en"],
        "legal_system": "Civil Law",
        "currency": "JPY"
    },
    "Germany": {
        "languages": ["de", "en"],
        "legal_system": "Civil Law",
        "currency": "EUR"
    },
    "France": {
        "languages": ["fr", "en"],
        "legal_system": "Civil Law", 
        "currency": "EUR"
    },
    "Canada": {
        "languages": ["en", "fr"],
        "legal_system": "Common Law",
        "currency": "CAD"
    },
    "Australia": {
        "languages": ["en"],
        "legal_system": "Common Law",
        "currency": "AUD"
    }
}

LANGUAGE_SUPPORT = {
    "en": {"name": "English", "voice": "en-US", "pdf": "en"},
    "hi": {"name": "हिन्दी (Hindi)", "voice": "hi-IN", "pdf": "hi"},
    "ta": {"name": "தமிழ் (Tamil)", "voice": "ta-IN", "pdf": "ta"},
    "te": {"name": "తెలుగు (Telugu)", "voice": "te-IN", "pdf": "te"},
    "ml": {"name": "മലയാളം (Malayalam)", "voice": "ml-IN", "pdf": "ml"},
    "kn": {"name": "ಕನ್ನಡ (Kannada)", "voice": "kn-IN", "pdf": "kn"},
    "es": {"name": "Español (Spanish)", "voice": "es-ES", "pdf": "es"},
    "fr": {"name": "Français (French)", "voice": "fr-FR", "pdf": "fr"},
    "de": {"name": "Deutsch (German)", "voice": "de-DE", "pdf": "de"},
    "ja": {"name": "日本語 (Japanese)", "voice": "ja-JP", "pdf": "ja"}
}

# --- PDF LAW DATABASE CREATOR ---
class LawDatabaseCreator:
    def __init__(self, db: Session):
        self.db = db
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with fitz.open(pdf_path) as doc:
                text = "\n".join([page.get_text() for page in doc])
            return text
        except Exception as e:
            print(f"❌ Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def process_laws_folder(self, laws_folder: str = "laws"):
        """Process all PDF files in laws folder and create database"""
        if not os.path.exists(laws_folder):
            print(f"⚠️ Laws folder '{laws_folder}' not found. Creating sample laws...")
            self.create_sample_laws()
            return
        
        pdf_files = glob.glob(os.path.join(laws_folder, "**", "*.pdf"), recursive=True)
        
        if not pdf_files:
            print("⚠️ No PDF files found in laws folder. Creating sample laws...")
            self.create_sample_laws()
            return
        
        print(f"📚 Found {len(pdf_files)} PDF files. Processing...")
        
        for pdf_file in pdf_files:
            country = os.path.basename(os.path.dirname(pdf_file))
            filename = os.path.basename(pdf_file)
            
            print(f"Processing: {country}/{filename}")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                continue
                
            # Create law entry
            law = Law(
                title=f"{country} Law - {os.path.splitext(filename)[0]}",
                jurisdiction=country,
                country=country,
                tags=f"{country}, legislation, statute",
                category="Legislation",
                text=text[:10000],  # Limit text length
                source_file=filename
            )
            
            self.db.add(law)
        
        self.db.commit()
        print(f"✅ Successfully processed {len(pdf_files)} law files into database")
    
    def create_sample_laws(self):
        """Create comprehensive sample laws for all countries"""
        print("📝 Creating comprehensive sample legal database...")
        
        sample_laws = [
            # Indian Laws
            {
                "title": "Indian Contract Act, 1872 - Essential Elements",
                "jurisdiction": "India",
                "country": "India",
                "tags": "contract, agreement, free consent, consideration",
                "category": "Contract Law",
                "text": "Essential elements of a valid contract: 1) Offer and acceptance, 2) Lawful consideration, 3) Capacity to contract, 4) Free consent, 5) Lawful object, 6) Not expressly declared void. Section 10: All agreements are contracts if made by free consent of parties competent to contract."
            },
            {
                "title": "Consumer Protection Act, 2019 - Consumer Rights",
                "jurisdiction": "India", 
                "country": "India",
                "tags": "consumer, rights, protection, unfair trade",
                "category": "Consumer Law",
                "text": "Rights of consumers: 1) Right to be protected against marketing of hazardous goods, 2) Right to be informed about quality and price, 3) Right to access variety at competitive prices, 4) Right to seek redressal against unfair practices, 5) Right to consumer education."
            },
            
            # USA Laws
            {
                "title": "Uniform Commercial Code (UCC) - Sales",
                "jurisdiction": "USA",
                "country": "USA", 
                "tags": "commercial, sales, goods, merchant",
                "category": "Commercial Law",
                "text": "UCC Article 2 governs sales of goods. Key provisions: 1) Goods must be movable at time of identification, 2) Merchant higher standards apply to professionals, 3) Statute of Frauds requires written contract for goods over $500, 4) Perfect tender rule requires exact performance."
            },
            {
                "title": "Americans with Disabilities Act (ADA)",
                "jurisdiction": "USA",
                "country": "USA",
                "tags": "disability, discrimination, accommodation, employment",
                "category": "Employment Law", 
                "text": "Prohibits discrimination against individuals with disabilities in employment, transportation, public accommodation. Requires reasonable accommodations unless undue hardship. Covers employers with 15+ employees. Protected disabilities include physical and mental impairments."
            },
            
            # UK Laws
            {
                "title": "UK Consumer Rights Act 2015",
                "jurisdiction": "UK",
                "country": "UK",
                "tags": "consumer, rights, goods, services, digital",
                "category": "Consumer Law",
                "text": "Goods must be: 1) Of satisfactory quality, 2) Fit for particular purpose, 3) As described. Services must be: 1) Performed with reasonable care and skill, 2) Within reasonable time if no fixed time agreed. 30-day right to reject faulty goods."
            },
            {
                "title": "UK Equality Act 2010",
                "jurisdiction": "UK", 
                "country": "UK",
                "tags": "equality, discrimination, protected characteristics",
                "category": "Employment Law",
                "text": "Protects against discrimination based on: age, disability, gender reassignment, marriage/civil partnership, pregnancy/maternity, race, religion/belief, sex, sexual orientation. Covers employment, education, housing, services. Direct and indirect discrimination prohibited."
            },
            
            # Japan Laws
            {
                "title": "Japanese Civil Code - Contracts",
                "jurisdiction": "Japan",
                "country": "Japan", 
                "tags": "civil code, contract, obligations, japan",
                "category": "Contract Law",
                "text": "A contract is formed by offer and acceptance. Parties must have intention to create legal relations. Consideration not required. Contracts can be oral or written. Good faith principle governs contract performance. Unilateral mistake may void contract if other party knew."
            },
            {
                "title": "Japanese Labor Standards Act",
                "jurisdiction": "Japan",
                "country": "Japan",
                "tags": "labor, employment, working hours, wages",
                "category": "Labor Law",
                "text": "Standard working hours: 8 hours/day, 40 hours/week. Overtime requires premium pay. Minimum wage set by region. Annual paid leave: 10 days after 6 months, increases with tenure. Discrimination based on nationality, creed, or social status prohibited."
            },
            
            # Germany Laws
            {
                "title": "German Civil Code (BGB) - Contracts",
                "jurisdiction": "Germany", 
                "country": "Germany",
                "tags": "civil code, contract, germany, BGB",
                "category": "Contract Law",
                "text": "Contracts require offer and acceptance. Principle of freedom of contract. Good faith (Treu und Glauben) is fundamental. Standard business terms regulated. Withdrawal rights for consumers. Contracts for work and services distinguished from sales contracts."
            },
            {
                "title": "German Commercial Code (HGB)",
                "jurisdiction": "Germany",
                "country": "Germany",
                "tags": "commercial, merchants, business, germany",
                "category": "Commercial Law", 
                "text": "Applies to merchants (Kaufleute). Higher standards for commercial transactions. Commercial letters must contain specific information. Statute of limitations for commercial claims. Special rules for commercial agents, brokers, and commercial partnerships."
            },
            
            # France Laws
            {
                "title": "French Civil Code - Contract Formation",
                "jurisdiction": "France",
                "country": "France",
                "tags": "civil code, contract, france, consent",
                "category": "Contract Law", 
                "text": "Four essential conditions for valid contract: 1) Consent of parties, 2) Capacity to contract, 3) Certain object, 4) Lawful cause. Consent vitiated by error, fraud, or duress. Contracts generally require proof in writing for values over €1500."
            },
            {
                "title": "French Consumer Code",
                "jurisdiction": "France",
                "country": "France",
                "tags": "consumer, protection, france, rights",
                "category": "Consumer Law",
                "text": "Right of withdrawal for distance contracts (14 days). Prohibition of abusive clauses. Mandatory pre-contractual information. Product liability rules. Price display requirements. Cooling-off periods for various contracts including doorstep selling."
            }
        ]
        
        for law_data in sample_laws:
            law = Law(**law_data)
            self.db.add(law)
        
        self.db.commit()
        print(f"✅ Created {len(sample_laws)} sample laws for global jurisdictions")

# --- DOCUMENT PROCESSING WITH OCR ---
class DocumentProcessor:
    def __init__(self):
        try:
            self.client = documentai.DocumentProcessorServiceClient()
            self.processor_name = DOC_AI_PROCESSOR_NAME
            print("✅ Document AI client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Document AI client: {e}")
            self.client = None
            self.processor_name = None

    def extract_text_with_document_ai(self, file_content: bytes, mime_type: str) -> str:
        """Extract text using Document AI OCR"""
        if not self.client or not self.processor_name:
            raise HTTPException(status_code=501, detail="Document AI service not configured")
        
        try:
            document = {"content": file_content, "mime_type": mime_type}
            result = self.client.process_document(
                request={"name": self.processor_name, "raw_document": document}
            )
            print("✅ Document AI OCR completed successfully")
            return result.document.text
        except Exception as e:
            print(f"Document AI processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

    def extract_text_from_file(self, file: UploadFile) -> str:
        """Extract text from various file types including images with OCR"""
        filename = file.filename or ""
        file_content = file.file.read()
        file.file.seek(0)
        
        mime_type = self._get_mime_type(filename)
        
        try:
            if mime_type == "application/pdf":
                return self._process_pdf(file_content, mime_type)
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self._process_docx(file_content)
            elif mime_type and mime_type.startswith("image/"):
                print(f"📷 Processing image with Document AI OCR: {mime_type}")
                return self.extract_text_with_document_ai(file_content, mime_type)
            elif mime_type == "text/plain":
                return file_content.decode("utf-8", errors="ignore")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

    def _get_mime_type(self, filename: str) -> str:
        mime_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.tiff': 'image/tiff',
            '.gif': 'image/gif', '.bmp': 'image/bmp'
        }
        ext = os.path.splitext(filename)[1].lower()
        return mime_types.get(ext, 'application/octet-stream')

    def _process_pdf(self, file_content: bytes, mime_type: str) -> str:
        """Process PDF with OCR fallback for scanned documents"""
        try:
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
            
            if len(text.strip()) < 100 or self._is_scanned_pdf(text):
                print("📄 PDF appears to be scanned, using Document AI OCR...")
                text = self.extract_text_with_document_ai(file_content, mime_type)
            return text
        except Exception as e:
            print(f"PDF processing failed, trying Document AI OCR: {e}")
            return self.extract_text_with_document_ai(file_content, mime_type)

    def _is_scanned_pdf(self, text: str) -> bool:
        clean_text = re.sub(r'\s+', '', text)
        return len(clean_text) < 50

    def _process_docx(self, file_content: bytes) -> str:
        """Process DOCX files"""
        try:
            doc = docx.Document(io.BytesIO(file_content))
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DOCX processing failed: {str(e)}")

# Initialize document processor
doc_processor = DocumentProcessor()

# --- LEGAL AI SERVICE ---
class LegalAIService:
    def __init__(self):
        self.model = model

    async def get_ai_response(self, prompt: str) -> str:
        """Get response from Gemini AI"""
        if not self.model:
            return "AI service not configured"
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"AI Error: {e}")
            return f"AI error: {e}"

    async def analyze_legal_document(self, document_text: str, language: str, country: str, legal_context: List[str]) -> Dict[str, Any]:
        """Comprehensive legal document analysis for specific country"""
        context = "\n".join(legal_context)
        
        prompt = f"""
        You are an experienced {country} legal attorney analyzing a legal document. 
        Provide comprehensive legal analysis in {LANGUAGE_SUPPORT[language]['name']}.

        JURISDICTION: {country}
        LEGAL SYSTEM: {COUNTRIES.get(country, {}).get('legal_system', 'Unknown')}
        
        RELEVANT {country.upper()} LAWS:
        {context}

        DOCUMENT TO ANALYZE:
        {document_text[:12000]}

        Provide your analysis in this EXACT JSON format:
        {{
            "executive_summary": "Overall legal assessment specific to {country}",
            "jurisdiction_analysis": "Analysis of {country} legal compliance",
            "key_provisions": ["List of key clauses identified"],
            "risk_analysis": {{
                "high_risk_items": [
                    {{
                        "clause": "exact clause text",
                        "risk_level": "HIGH",
                        "legal_issue": "specific legal problem under {country} law",
                        "statutory_violation": "which {country} law is violated",
                        "potential_consequence": "legal/financial impact in {country}", 
                        "remedial_action": "specific legal remedy for {country}"
                    }}
                ],
                "moderate_risk_items": [
                    {{
                        "clause": "exact clause text", 
                        "risk_level": "MODERATE",
                        "legal_issue": "specific concern under {country} law",
                        "recommendation": "suggested modification for {country} compliance"
                    }}
                ]
            }},
            "legal_recommendations": [
                "Specific legal advice for {country} jurisdiction"
            ],
            "compliance_score": 0-100,
            "negotiation_strategy": "Detailed legal negotiation approach for {country}"
        }}

        Focus on {country} specific:
        1. Contract validity and enforceability under {country} law
        2. Statutory compliance with {country} regulations
        3. Consumer protection under {country} law
        4. Employment standards if applicable
        5. Industry-specific regulations in {country}
        """
        
        response = await self.get_ai_response(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not parse legal analysis"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in legal analysis"}

# Initialize AI service
legal_ai = LegalAIService()

# --- LEGAL RESEARCH SERVICE ---
class LegalResearchService:
    def __init__(self, db: Session):
        self.db = db

    async def find_relevant_laws(self, document_text: str, country: str = "India") -> List[str]:
        """Find legally relevant statutes for specific country"""
        # Extract key legal terms
        legal_terms = self._extract_legal_terms(document_text)
        
        relevant_laws = []
        for term in legal_terms:
            laws = self.db.query(Law).filter(
                (Law.country == country) &
                ((Law.text.ilike(f"%{term}%")) | 
                 (Law.tags.ilike(f"%{term}%")) |
                 (Law.title.ilike(f"%{term}%")))
            ).limit(5).all()
            relevant_laws.extend(laws)
        
        # Remove duplicates
        seen = set()
        unique_laws = []
        for law in relevant_laws:
            if law.id not in seen:
                seen.add(law.id)
                unique_laws.append(law)
        
        return [f"{law.title} ({law.category}): {law.text}" for law in unique_laws[:8]]

    def _extract_legal_terms(self, text: str) -> List[str]:
        """Extract potential legal terms from document"""
        legal_keywords = [
            'contract', 'agreement', 'liability', 'indemnity', 'termination',
            'breach', 'damages', 'jurisdiction', 'arbitration', 'confidentiality',
            'warranty', 'guarantee', 'insurance', 'compensation', 'penalty',
            'rent', 'deposit', 'lease', 'tenancy', 'employment', 'salary',
            'termination', 'notice', 'intellectual property', 'copyright',
            'data protection', 'privacy', 'consent', 'consumer', 'rights'
        ]
        
        found_terms = []
        text_lower = text.lower()
        for term in legal_keywords:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms

# --- PDF REPORT GENERATION ---
class LegalReportPDF(FPDF):
    def __init__(self, language="en", country="India"):
        super().__init__()
        self.language = language
        self.country = country
        
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, f'LEGAL ANALYSIS REPORT - {self.country.upper()}', 0, 1, 'C')
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def add_legal_section(self, title, content):
        self.add_page()
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, title, 0, 1)
        self.ln(2)
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, content)
        self.ln(5)

# --- SCHEMAS ---
class AskRequest(BaseModel):
    document_text: str
    question: str
    language: str = "en"
    country: str = "India"

class AskResponse(BaseModel):
    answer: str

class NegotiateRequest(BaseModel):
    history: list
    user_message: str
    language: str = "en"
    country: str = "India"

class NegotiateResponse(BaseModel):
    ai_response: str
    updated_history: list

class AnalyzeRequest(BaseModel):
    user_name: str = ""
    language: str = "en"
    country: str = "India"

class AnalyzeJobResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    result: Optional[Dict[str, Any]] = None

class DownloadReportRequest(BaseModel):
    analysis_result: Dict[str, Any]
    language: str = "en"
    country: str = "India"

# --- DATABASE UTILITIES ---
def create_analysis_job(db: Session, job_id: str, file_path: str, user_name: str, language: str, country: str) -> AnalysisJob:
    """Create new analysis job"""
    job = AnalysisJob(
        job_id=job_id,
        file_path=file_path,
        user_name=user_name,
        language=language,
        country=country,
        status=AnalysisStatus.PENDING
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job

def get_analysis_job(db: Session, job_id: str) -> Optional[AnalysisJob]:
    """Get analysis job by ID"""
    return db.query(AnalysisJob).filter(AnalysisJob.job_id == job_id).first()

# --- ASYNC PROCESSING ---
async def process_legal_analysis(job_id: str, document_text: str, language: str, country: str, db: Session):
    """Process legal document analysis asynchronously"""
    job = get_analysis_job(db, job_id)
    if not job:
        return

    job.status = AnalysisStatus.PROCESSING
    db.commit()

    try:
        # Perform legal research for specific country
        research_service = LegalResearchService(db)
        relevant_laws = await research_service.find_relevant_laws(document_text, country)
        
        # Perform comprehensive legal analysis for country
        analysis_result = await legal_ai.analyze_legal_document(
            document_text, language, country, relevant_laws
        )
        
        # Save results
        job.analysis_result_json = json.dumps(analysis_result)
        job.status = AnalysisStatus.COMPLETE
        db.commit()
        
        print(f"✅ Legal analysis completed for {country} - job {job_id}")
        
    except Exception as e:
        job.status = AnalysisStatus.FAILED
        db.commit()
        print(f"❌ Legal analysis failed for {country} - job {job_id}: {e}")

# --- FRONTEND HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Global Legal AI Demystifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0F0F1A;
            --primary-accent: #8A2BE2;
            --secondary-accent: #00BFFF;
            --glow-color: rgba(138, 43, 226, 0.6);
            --text-color: #EAEAEA;
            --text-muted: #A0A0A0;
            --font-family: 'Poppins', sans-serif;
            --card-bg: rgba(22, 22, 34, 0.6);
            --high-risk-bg: rgba(233, 69, 96, 0.25);
            --caution-bg: rgba(247, 183, 49, 0.25);
            --safe-bg: rgba(46, 204, 113, 0.25);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-family); background-color: var(--bg-color);
            color: var(--text-color); padding: 20px;
            line-height: 1.7; overflow-x: hidden;
        }
        
        #particle-canvas {
            position: fixed; top: 0; left: 0;
            z-index: -1; width: 100%; height: 100%;
        }
        
        #splash-screen {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: var(--bg-color); display: flex; flex-direction: column;
            justify-content: center; align-items: center; z-index: 1000;
            transition: opacity 0.5s ease-out;
        }
        
        #splash-screen .logo { font-size: 5rem; animation: pulse 2s infinite; }
        
        .container { max-width: 1800px; margin: 0 auto; }
        
        header { text-align: center; margin-bottom: 20px; animation: fadeInDown 1s ease-out; }
        
        h1 {
            background: linear-gradient(90deg, var(--primary-accent), var(--secondary-accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700; font-size: 2.8rem; margin-bottom: 10px;
        }
        
        .settings-bar { 
            display: flex; 
            gap: 15px; 
            justify-content: center; 
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .settings-bar select {
            padding: 10px 15px; border-radius: 8px; border: 1px solid var(--primary-accent);
            background-color: var(--card-bg); color: var(--text-color); font-size: 1rem;
            cursor: pointer; transition: box-shadow 0.3s ease;
        }
        
        .card, .document-input-card {
            background-color: var(--card-bg); backdrop-filter: blur(10px); 
            border-radius: 20px; padding: 30px; margin-bottom: 25px;
            border: 1px solid rgba(138, 43, 226, 0.3);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.3s ease, border-color 0.3s ease;
            animation: card-fade-in 0.6s ease-out forwards;
        }
        
        .card-glow:hover { transform: translateY(-8px); border-color: rgba(138, 43, 226, 0.8); }
        
        .file-upload-label {
            display: block; padding: 20px; border-radius: 10px;
            border: 2px dashed rgba(138, 43, 226, 0.5); background-color: rgba(0,0,0,0.2);
            color: var(--text-muted); text-align: center; cursor: pointer;
            transition: background-color 0.3s, border-color 0.3s;
        }
        
        .file-upload-label:hover { border-color: var(--secondary-accent); color: #fff; }
        
        .file-name-display { display: block; text-align: center; margin-top: 10px; color: var(--secondary-accent); }
        
        input[type="text"], textarea {
            width: 100%; padding: 15px; border-radius: 10px;
            border: 1px solid rgba(138, 43, 226, 0.5); background-color: rgba(0,0,0,0.2);
            color: var(--text-color); font-size: 1rem; resize: vertical;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        
        input:focus, textarea:focus { outline: none; border-color: var(--secondary-accent); box-shadow: 0 0 15px var(--glow-color); }
        
        .autofill-section { display: flex; gap: 15px; align-items: center; margin-top: 15px;}
        
        button {
            background: linear-gradient(90deg, var(--primary-accent), var(--secondary-accent));
            color: white; padding: 15px 30px; border: none; border-radius: 10px; cursor: pointer;
            font-weight: 600; font-size: 1rem; transition: all 0.3s ease;
        }
        
        button:hover { transform: scale(1.05); box-shadow: 0 5px 20px var(--glow-color); }
        
        .dashboard-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding: 0 10px; }
        
        .dashboard-header h2 { color: #fff; }
        
        #downloadReportBtn { background: linear-gradient(90deg, #3f9a76, #2a6f53); }
        
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 30px; }
        
        .full-span-card { grid-column: 1 / -1; }
        
        .card-header { margin-top: 0; border-bottom: 1px solid rgba(138, 43, 226, 0.5); padding-bottom: 15px; font-size: 1.4rem; color: #fff; }
        
        .chat-box { height: 300px; overflow-y: auto; background-color: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px; }
        
        .qa-input { display: flex; gap: 10px; margin-top: 15px; }
        
        .user-msg, .ai-msg { padding: 10px 15px; border-radius: 18px; margin-bottom: 10px; max-width: 85%; animation: popIn 0.3s ease-out; }
        
        .user-msg { background: var(--primary-accent); align-self: flex-end; }
        .ai-msg { background: #2c3e50; align-self: flex-start; }
        
        .risk-item { padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 5px solid; transition: transform 0.3s; }
        
        .risk-high { border-color: #e94560; background-color: var(--high-risk-bg); }
        .risk-caution { border-color: #f7b731; background-color: var(--caution-bg); }
        .risk-safe { border-color: #2ecc71; background-color: var(--safe-bg); }
        
        .code-box, .result-box { margin-top: 15px; padding: 15px; background-color: rgba(0,0,0,0.2); border-radius: 10px; white-space: pre-wrap; }
        
        .result-box ul { padding-left: 20px; }
        
        footer { text-align: center; margin-top: 40px; color: var(--text-muted); font-size: 0.9rem; }
        
        .hidden { display: none !important; }
        
        #loader { border: 8px solid var(--primary-accent); border-top: 8px solid var(--secondary-accent); border-radius: 50%; width: 60px; height: 60px; animation: spin 1s linear infinite; margin: 40px auto; }
        
        @keyframes spin { 100% { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
        @keyframes card-fade-in { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        
        .status-badge { display: inline-block; padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }
        .status-pending { background: #fff3cd; color: #856404; }
        .status-processing { background: #cce7ff; color: #004085; }
        .status-complete { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }
        
        .legal-citation { font-style: italic; color: var(--secondary-accent); margin: 5px 0; }
        .legal-warning { color: #e94560; font-weight: bold; }
        .legal-recommendation { color: #2ecc71; font-weight: bold; }
        
        .country-flag { 
            display: inline-block; 
            width: 20px; 
            height: 15px; 
            margin-right: 8px;
            background-size: cover;
            border-radius: 2px;
            vertical-align: middle;
        }
    </style>
</head>
<body>
    <canvas id="particle-canvas"></canvas>
    <div id="splash-screen">
        <div class="logo">⚖️</div>
        <h1>Global Legal AI Demystifier</h1>
        <p>Multinational Legal Document Analysis</p>
    </div>

    <main id="app-body" class="hidden">
        <header>
            <h1>Global Legal AI Dashboard</h1>
            <p>Analyze Legal Documents Across Multiple Jurisdictions</p>
             <div class="settings-bar">
                 <select id="countrySelector">
                     <option value="India">🇮🇳 India</option>
                     <option value="USA">🇺🇸 United States</option>
                     <option value="UK">🇬🇧 United Kingdom</option>
                     <option value="Japan">🇯🇵 Japan</option>
                     <option value="Germany">🇩🇪 Germany</option>
                     <option value="France">🇫🇷 France</option>
                     <option value="Canada">🇨🇦 Canada</option>
                     <option value="Australia">🇦🇺 Australia</option>
                 </select>
                 
                 <select id="languageSelector">
                     <option value="en">English</option>
                     <option value="hi">हिन्दी (Hindi)</option>
                     <option value="ta">தமிழ் (Tamil)</option>
                     <option value="te">తెలుగు (Telugu)</option>
                     <option value="ml">മലയാളം (Malayalam)</option>
                     <option value="kn">ಕನ್ನಡ (Kannada)</option>
                     <option value="es">Español (Spanish)</option>
                     <option value="fr">Français (French)</option>
                     <option value="de">Deutsch (German)</option>
                     <option value="ja">日本語 (Japanese)</option>
                 </select>
             </div>
        </header>

        <div class="container">
            <div class="document-input-card card-glow">
                <h2>1. Begin Your Global Legal Analysis</h2>
                <div class="file-upload-area">
                    <input type="file" id="documentUpload" accept=".txt,.pdf,.docx,.jpg,.jpeg,.png" class="hidden-file-input">
                    <label for="documentUpload" class="file-upload-label">Click to Upload Legal Document (.txt, .pdf, .docx, Images)</label>
                    <span id="fileName" class="file-name-display"></span>
                </div>
                <div class="autofill-section">
                    <input type="text" id="autofillName" placeholder="Enter Your Full Name for Documentation">
                    <button id="analyzeBtn">🚀 Analyze & Build Legal Dashboard</button>
                </div>
                <div id="jurisdictionInfo" style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <strong>Selected Jurisdiction:</strong> <span id="currentCountry">India</span> - 
                    <span id="legalSystem">Common Law</span> - 
                    <span id="supportedLanguages">English, Hindi, Tamil, Telugu, Malayalam, Kannada</span>
                </div>
            </div>

            <div id="loader" class="hidden"></div>

            <div id="dashboard" class="hidden">
                 <div class="dashboard-header">
                    <h2>Legal Analysis Results - <span id="resultsCountry">India</span></h2>
                    <button id="downloadReportBtn">📄 Download Legal Report</button>
                </div>
                <div class="dashboard-grid">
                    <div class="card card-glow key-facts">
                        <h3 class="card-header">📊 Executive Legal Summary</h3>
                        <div id="executiveSummary" class="card-content"></div>
                    </div>
                    <div class="card card-glow risk-analysis">
                        <h3 class="card-header">🚦 Legal Risk Analysis</h3>
                        <div id="riskAnalysisOutput" class="card-content"></div>
                    </div>
                    <div class="card card-glow legal-recommendations">
                        <h3 class="card-header">💡 Legal Recommendations</h3>
                        <div id="legalRecommendations" class="card-content"></div>
                    </div>
                    <div class="card card-glow interactive-qa">
                        <h3 class="card-header">💬 Legal Consultation (Voice Enabled)</h3>
                        <div id="qaOutput" class="card-content chat-box"></div>
                        <div class="qa-input">
                            <input type="text" id="qaInput" placeholder="Ask a legal question...">
                            <button id="askBtn" title="Send Question">➤</button>
                            <button id="speakBtn" title="Ask with your voice">🎙️</button>
                        </div>
                    </div>
                    <div class="card card-glow negotiation-simulator">
                        <h3 class="card-header">🤝 Legal Negotiation Simulator</h3>
                        <div id="negotiationOutput" class="card-content chat-box"></div>
                        <div class="qa-input">
                            <input type="text" id="negotiationInput" placeholder="Type your negotiation point...">
                            <button id="negotiateBtn" title="Send Negotiation Point">➤</button>
                        </div>
                    </div>
                    <div class="card card-glow full-span-card compliance-score">
                        <h3 class="card-header">✅ Legal Compliance Score & Strategy</h3>
                        <div id="complianceScore" class="card-content"></div>
                    </div>
                </div>
            </div>
        </div>
    </main>
    <footer>
        <p>Built by Visoneers. For updates or changes, contact: <a href="mailto:sanjaymurugadoss02@gmail.com">sanjaymurugadoss02@gmail.com</a></p>
        <p>Disclaimer: This AI tool is for informational purposes only and is not a substitute for professional legal advice.</p>
    </footer>
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // Country and language data
            const countries = {
                "India": { legalSystem: "Common Law", languages: "English, Hindi, Tamil, Telugu, Malayalam, Kannada" },
                "USA": { legalSystem: "Common Law", languages: "English" },
                "UK": { legalSystem: "Common Law", languages: "English" },
                "Japan": { legalSystem: "Civil Law", languages: "Japanese, English" },
                "Germany": { legalSystem: "Civil Law", languages: "German, English" },
                "France": { legalSystem: "Civil Law", languages: "French, English" },
                "Canada": { legalSystem: "Common Law", languages: "English, French" },
                "Australia": { legalSystem: "Common Law", languages: "English" }
            };

            const voiceLanguages = {
                'en': 'en-US', 'hi': 'hi-IN', 'ta': 'ta-IN', 'te': 'te-IN',
                'ml': 'ml-IN', 'kn': 'kn-IN', 'es': 'es-ES', 'fr': 'fr-FR',
                'de': 'de-DE', 'ja': 'ja-JP'
            };

            // --- Initialize UI ---
            function updateJurisdictionInfo() {
                const country = document.getElementById('countrySelector').value;
                const countryInfo = countries[country];
                document.getElementById('currentCountry').textContent = country;
                document.getElementById('legalSystem').textContent = countryInfo.legalSystem;
                document.getElementById('supportedLanguages').textContent = countryInfo.languages;
                document.getElementById('resultsCountry').textContent = country;
            }

            // Set up country selector
            document.getElementById('countrySelector').addEventListener('change', updateJurisdictionInfo);
            updateJurisdictionInfo(); // Initial call

            // --- Particle Background ---
            const canvas = document.getElementById('particle-canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            let particles = [];
            function initParticles() {
                particles = [];
                for (let i = 0; i < 50; i++) {
                    particles.push({ x: Math.random() * canvas.width, y: Math.random() * canvas.height, size: Math.random() * 2 + 1, speedX: Math.random() * 1 - 0.5, speedY: Math.random() * 1 - 0.5, color: `rgba(138, 43, 226, ${Math.random()})` });
                }
            }
            function animateParticles() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                for (const p of particles) {
                    if (p.x > canvas.width || p.x < 0) p.speedX *= -1;
                    if (p.y > canvas.height || p.y < 0) p.speedY *= -1;
                    p.x += p.speedX; p.y += p.speedY;
                    ctx.fillStyle = p.color;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                    ctx.fill();
                }
                requestAnimationFrame(animateParticles);
            }
            initParticles();
            animateParticles();
            window.addEventListener('resize', () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; initParticles(); });

            // --- Splash Screen ---
            const splash = document.getElementById('splash-screen');
            setTimeout(() => {
                splash.style.opacity = '0';
                document.getElementById('app-body').classList.remove('hidden');
                setTimeout(() => splash.classList.add('hidden'), 500);
            }, 1500);

            // --- Global State ---
            let currentDocumentText = '', negotiationHistory = [], currentAnalysisResult = null;
            const langSelector = document.getElementById('languageSelector');
            const countrySelector = document.getElementById('countrySelector');

            // --- Helper Functions ---
            async function fetchAPI(endpoint, body, method = 'POST') {
                try {
                    const options = {
                        method: method,
                        headers: method !== 'GET' ? { 'Content-Type': 'application/json' } : {},
                        body: method !== 'GET' ? JSON.stringify(body) : undefined
                    };
                    
                    const response = await fetch(endpoint, options);
                    if (!response.ok) throw new Error('API call failed');
                    return await response.json();
                } catch (error) {
                    console.error('API Error:', error);
                    alert('Error: ' + error.message);
                    return null;
                }
            }

            function textToSpeech(text) {
                if (!('speechSynthesis' in window)) return;
                speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = voiceLanguages[langSelector.value] || 'en-US';
                window.speechSynthesis.speak(utterance);
            }

            // --- File Upload ---
            document.getElementById('documentUpload').addEventListener('change', (e) => {
                const file = e.target.files[0];
                document.getElementById('fileName').textContent = file ? file.name : '';
            });

            // --- Document Analysis ---
            document.getElementById('analyzeBtn').addEventListener('click', async () => {
                const file = document.getElementById('documentUpload').files[0];
                if (!file) { alert("Please upload a legal document."); return; }
                
                document.getElementById('loader').classList.remove('hidden');
                document.getElementById('dashboard').classList.add('hidden');
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('user_name', document.getElementById('autofillName').value);
                formData.append('language', langSelector.value);
                formData.append('country', countrySelector.value);

                try {
                    const response = await fetch('/analyze_async', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    if (result.job_id) {
                        currentJobId = result.job_id;
                        pollJobStatus();
                    }
                } catch (error) {
                    alert('Analysis failed: ' + error.message);
                    document.getElementById('loader').classList.add('hidden');
                }
            });

            let currentJobId = null;
            async function pollJobStatus() {
                if (!currentJobId) return;
                
                const checkInterval = setInterval(async () => {
                    const result = await fetchAPI(`/analyze/status/${currentJobId}`, {}, 'GET');
                    if (result) {
                        updateJobStatus(result);
                        if (result.status === 'COMPLETE' || result.status === 'FAILED') {
                            clearInterval(checkInterval);
                            document.getElementById('loader').classList.add('hidden');
                        }
                    }
                }, 2000);
            }

            function updateJobStatus(statusData) {
                if (statusData.status === 'COMPLETE' && statusData.result) {
                    currentAnalysisResult = statusData.result;
                    displayLegalAnalysis(statusData.result);
                } else if (statusData.status === 'FAILED') {
                    document.getElementById('executiveSummary').innerHTML = 
                        '<div class="risk-item risk-high">Legal analysis failed. Please try again.</div>';
                }
            }

            function displayLegalAnalysis(analysis) {
                if (analysis.error) {
                    document.getElementById('executiveSummary').innerHTML = 
                        '<div class="risk-item risk-high">Error: ' + analysis.error + '</div>';
                    return;
                }

                // Executive Summary
                let html = `<p>${analysis.executive_summary || 'No executive summary available.'}</p>`;
                if (analysis.jurisdiction_analysis) {
                    html += `<p><strong>Jurisdiction Analysis:</strong> ${analysis.jurisdiction_analysis}</p>`;
                }
                document.getElementById('executiveSummary').innerHTML = html;

                // Risk Analysis
                let riskHtml = '';
                if (analysis.risk_analysis) {
                    if (analysis.risk_analysis.high_risk_items && analysis.risk_analysis.high_risk_items.length > 0) {
                        riskHtml += '<h4>🚨 High Risk Items</h4>';
                        analysis.risk_analysis.high_risk_items.forEach(item => {
                            riskHtml += `
                                <div class="risk-item risk-high">
                                    <strong>${item.clause}</strong><br>
                                    <span class="legal-warning">Legal Issue: ${item.legal_issue}</span><br>
                                    <span class="legal-citation">Statutory Violation: ${item.statutory_violation}</span><br>
                                    <em>Potential Consequence: ${item.potential_consequence}</em><br>
                                    <span class="legal-recommendation">Remedial Action: ${item.remedial_action}</span>
                                </div>
                            `;
                        });
                    }

                    if (analysis.risk_analysis.moderate_risk_items && analysis.risk_analysis.moderate_risk_items.length > 0) {
                        riskHtml += '<h4>⚠️ Moderate Risk Items</h4>';
                        analysis.risk_analysis.moderate_risk_items.forEach(item => {
                            riskHtml += `
                                <div class="risk-item risk-caution">
                                    <strong>${item.clause}</strong><br>
                                    <span class="legal-warning">Legal Concern: ${item.legal_issue}</span><br>
                                    <span class="legal-recommendation">Recommendation: ${item.recommendation}</span>
                                </div>
                            `;
                        });
                    }
                }
                document.getElementById('riskAnalysisOutput').innerHTML = riskHtml || '<p>No significant risks identified.</p>';

                // Legal Recommendations
                let recHtml = '';
                if (analysis.legal_recommendations && analysis.legal_recommendations.length > 0) {
                    recHtml += '<ul>';
                    analysis.legal_recommendations.forEach(rec => {
                        recHtml += `<li class="legal-recommendation">${rec}</li>`;
                    });
                    recHtml += '</ul>';
                }
                document.getElementById('legalRecommendations').innerHTML = recHtml || '<p>No specific recommendations available.</p>';

                // Compliance Score
                let scoreHtml = '';
                if (analysis.compliance_score !== undefined) {
                    const score = analysis.compliance_score;
                    const scoreClass = score >= 80 ? 'risk-safe' : score >= 60 ? 'risk-caution' : 'risk-high';
                    scoreHtml += `
                        <div class="risk-item ${scoreClass}">
                            <h3>Legal Compliance Score: ${score}/100</h3>
                            ${analysis.negotiation_strategy ? `<p><strong>Negotiation Strategy:</strong> ${analysis.negotiation_strategy}</p>` : ''}
                        </div>
                    `;
                }
                document.getElementById('complianceScore').innerHTML = scoreHtml || '<p>Compliance score not available.</p>';

                document.getElementById('dashboard').classList.remove('hidden');
            }

            // --- Legal Consultation ---
            document.getElementById('askBtn').addEventListener('click', sendLegalQuestion);
            document.getElementById('qaInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendLegalQuestion();
            });

            async function sendLegalQuestion() {
                const input = document.getElementById('qaInput');
                const question = input.value.trim();
                if (!question) return;

                const qaOutput = document.getElementById('qaOutput');
                qaOutput.innerHTML += `<div class="user-msg">${question}</div>`;
                qaOutput.scrollTop = qaOutput.scrollHeight;
                input.value = '';

                const result = await fetchAPI('/ask_legal_question', {
                    document_text: currentDocumentText,
                    question: question,
                    language: langSelector.value,
                    country: countrySelector.value
                });

                if (result) {
                    const aiMsg = document.createElement('div');
                    aiMsg.className = 'ai-msg';
                    aiMsg.innerHTML = result.answer;
                    qaOutput.appendChild(aiMsg);
                    qaOutput.scrollTop = qaOutput.scrollHeight;
                    textToSpeech(result.answer);
                }
            }

            // --- Voice Input ---
            document.getElementById('speakBtn').addEventListener('click', startVoiceInput);

            function startVoiceInput() {
                if (!('webkitSpeechRecognition' in window)) {
                    alert('Speech recognition not supported in this browser');
                    return;
                }

                const recognition = new webkitSpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = voiceLanguages[langSelector.value] || 'en-US';

                recognition.onstart = () => {
                    document.getElementById('qaInput').placeholder = 'Listening...';
                };

                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    document.getElementById('qaInput').value = transcript;
                    document.getElementById('qaInput').placeholder = 'Ask a legal question...';
                };

                recognition.onend = () => {
                    document.getElementById('qaInput').placeholder = 'Ask a legal question...';
                };

                recognition.start();
            }

            // --- Negotiation Simulator ---
            document.getElementById('negotiateBtn').addEventListener('click', sendNegotiation);
            document.getElementById('negotiationInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendNegotiation();
            });

            async function sendNegotiation() {
                const input = document.getElementById('negotiationInput');
                const message = input.value.trim();
                if (!message) return;

                const negOutput = document.getElementById('negotiationOutput');
                negOutput.innerHTML += `<div class="user-msg">${message}</div>`;
                negOutput.scrollTop = negOutput.scrollHeight;
                input.value = '';

                const result = await fetchAPI('/negotiate', {
                    history: negotiationHistory,
                    user_message: message,
                    language: langSelector.value,
                    country: countrySelector.value
                });

                if (result) {
                    negotiationHistory = result.updated_history;
                    const aiMsg = document.createElement('div');
                    aiMsg.className = 'ai-msg';
                    aiMsg.innerHTML = result.ai_response;
                    negOutput.appendChild(aiMsg);
                    negOutput.scrollTop = negOutput.scrollHeight;
                }
            }

            // --- Report Download ---
            document.getElementById('downloadReportBtn').addEventListener('click', async () => {
                if (!currentAnalysisResult) {
                    alert('Please analyze a document first');
                    return;
                }

                const result = await fetchAPI('/download_legal_report', {
                    analysis_result: currentAnalysisResult,
                    language: langSelector.value,
                    country: countrySelector.value
                });

                if (result && result.download_url) {
                    window.open(result.download_url, '_blank');
                }
            });
        });
    </script>
</body>
</html>
"""

# --- FASTAPI APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Global Legal AI Demystifier...")
    db = SessionLocal()
    try:
        # Create law database from PDF files or create sample laws
        db_creator = LawDatabaseCreator(db)
        db_creator.process_laws_folder("laws")
        print(f"📊 Database contains {db.query(Law).count()} legal provisions")
    finally:
        db.close()
    print("✅ Application startup complete.")
    yield
    print("🌙 Application shutting down.")

app = FastAPI(title="Global Legal AI Demystifier", lifespan=lifespan)

# --- API ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.post("/analyze_async", response_model=AnalyzeJobResponse)
async def analyze_document_async(
    background_tasks: BackgroundTasks,
    user_name: str = Form(""),
    language: str = Form("en"),
    country: str = Form("India"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Start async legal document analysis for specific country"""
    try:
        print(f"📄 Processing legal document for {country}: {file.filename}")
        
        # Extract text with OCR support
        document_text = doc_processor.extract_text_from_file(file)
        
        print(f"✅ Text extracted. Length: {len(document_text)} characters")
        
        # Create job
        job_id = f"job_{uuid.uuid4()}"
        job = create_analysis_job(db, job_id, file.filename, user_name, language, country)
        
        # Start async legal analysis
        background_tasks.add_task(process_legal_analysis, job_id, document_text, language, country, db)
        
        return AnalyzeJobResponse(
            job_id=job.job_id,
            status=job.status,
            message=f"Legal analysis started for {country}. This may take a few moments."
        )
    except Exception as e:
        print(f"❌ Legal analysis failed for {country}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Legal analysis failed: {str(e)}")

@app.get("/analyze/status/{job_id}", response_model=JobStatusResponse)
async def get_analysis_status(job_id: str, db: Session = Depends(get_db)):
    """Get analysis job status"""
    job = get_analysis_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    
    result = None
    if job.analysis_result_json:
        try:
            result = json.loads(job.analysis_result_json)
        except json.JSONDecodeError:
            result = {"error": "Failed to parse legal analysis results"}
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=result
    )

@app.post("/ask_legal_question", response_model=AskResponse)
async def ask_legal_question(req: AskRequest, db: Session = Depends(get_db)):
    """Get professional legal advice for specific country"""
    # Find relevant laws for country context
    research_service = LegalResearchService(db)
    relevant_laws = await research_service.find_relevant_laws(req.document_text, req.country)
    legal_context = "\n".join(relevant_laws)
    
    prompt = f"""
    As an experienced {req.country} legal attorney, provide professional legal advice in {LANGUAGE_SUPPORT[req.language]['name']}.

    JURISDICTION: {req.country}
    LEGAL CONTEXT:
    {legal_context}

    CLIENT QUESTION:
    {req.question}

    Provide comprehensive legal advice specific to {req.country} law covering:
    1. Legal position based on {req.country} statutes
    2. Potential risks and liabilities under {req.country} law
    3. Recommended course of action for {req.country} jurisdiction
    4. {req.country} legal precedents if applicable
    5. Next steps for the client in {req.country}

    Structure your response professionally as a {req.country} legal counsel would advise a client.
    """
    
    answer = await legal_ai.get_ai_response(prompt)
    return AskResponse(answer=answer)

@app.post("/negotiate", response_model=NegotiateResponse)
async def negotiate(req: NegotiateRequest):
    """Legal negotiation simulator for specific country"""
    prompt = f"""
    You are a {req.country} legal negotiation expert. Continue this legal negotiation in {LANGUAGE_SUPPORT[req.language]['name']}.
    
    JURISDICTION: {req.country}
    LEGAL SYSTEM: {COUNTRIES.get(req.country, {}).get('legal_system', 'Unknown')}
    
    Negotiation History:
    {json.dumps(req.history[-5:], ensure_ascii=False)}
    
    Client's latest point: {req.user_message}
    
    Respond as a professional {req.country} legal negotiator would, considering:
    - {req.country} legal positions and statutory requirements
    - Risk mitigation strategies under {req.country} law
    - Practical compromise solutions for {req.country} jurisdiction
    - {req.country} legal precedent where applicable
    """
    
    response = await legal_ai.get_ai_response(prompt)
    
    updated_history = req.history + [
        {'role': 'user', 'parts': [req.user_message]},
        {'role': 'model', 'parts': [response]}
    ]
    
    return NegotiateResponse(
        ai_response=response,
        updated_history=updated_history
    )

@app.post("/download_legal_report")
async def download_legal_report(req: DownloadReportRequest):
    """Generate and download comprehensive legal report for country"""
    try:
        # Create PDF report
        pdf = LegalReportPDF(req.language, req.country)
        
        # Add sections
        analysis = req.analysis_result
        
        if 'executive_summary' in analysis:
            pdf.add_legal_section("EXECUTIVE LEGAL SUMMARY", analysis['executive_summary'])
        
        if 'jurisdiction_analysis' in analysis:
            pdf.add_legal_section(f"{req.country.upper()} JURISDICTION ANALYSIS", analysis['jurisdiction_analysis'])
        
        if 'risk_analysis' in analysis:
            risk_content = ""
            if analysis['risk_analysis'].get('high_risk_items'):
                risk_content += "HIGH RISK ITEMS:\n\n"
                for item in analysis['risk_analysis']['high_risk_items']:
                    risk_content += f"Clause: {item.get('clause', 'N/A')}\n"
                    risk_content += f"Legal Issue: {item.get('legal_issue', 'N/A')}\n"
                    risk_content += f"Statutory Violation: {item.get('statutory_violation', 'N/A')}\n"
                    risk_content += f"Potential Consequence: {item.get('potential_consequence', 'N/A')}\n"
                    risk_content += f"Remedial Action: {item.get('remedial_action', 'N/A')}\n\n"
            
            if analysis['risk_analysis'].get('moderate_risk_items'):
                risk_content += "MODERATE RISK ITEMS:\n\n"
                for item in analysis['risk_analysis']['moderate_risk_items']:
                    risk_content += f"Clause: {item.get('clause', 'N/A')}\n"
                    risk_content += f"Legal Concern: {item.get('legal_issue', 'N/A')}\n"
                    risk_content += f"Recommendation: {item.get('recommendation', 'N/A')}\n\n"
            
            pdf.add_legal_section("LEGAL RISK ANALYSIS", risk_content)
        
        if 'legal_recommendations' in analysis:
            rec_content = "\n".join([f"• {rec}" for rec in analysis['legal_recommendations']])
            pdf.add_legal_section("LEGAL RECOMMENDATIONS", rec_content)
        
        if 'compliance_score' in analysis:
            score_content = f"Overall Legal Compliance Score for {req.country}: {analysis['compliance_score']}/100\n\n"
            if 'negotiation_strategy' in analysis:
                score_content += f"Recommended Negotiation Strategy for {req.country}:\n{analysis['negotiation_strategy']}"
            pdf.add_legal_section("COMPLIANCE ASSESSMENT", score_content)
        
        # Generate PDF
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={req.country}_legal_analysis_report.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# --- RUN APPLICATION ---
if __name__ == "__main__":
    print("🚀 Starting Global Legal AI Demystifier on http://localhost:8000")
    print("🌍 Supported Countries: India, USA, UK, Japan, Germany, France, Canada, Australia")
    print("🗣️ Supported Languages: English, Hindi, Tamil, Telugu, Malayalam, Kannada, Spanish, French, German, Japanese")
    print("📚 Features: Automatic PDF Law Database, Country Selection, OCR, Multilingual Voice, Professional Reports")
    uvicorn.run(app, host="0.0.0.0", port=8000)