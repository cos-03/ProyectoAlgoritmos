from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os

from academic_analysis_gui import AcademicAnalysisAPI

# Modelos Pydantic para validación
class LoginRequest(BaseModel):
    databases: List[str]
    email: str
    password: str
    show_browser: bool = False

class ScrapingRequest(BaseModel):
    query: str
    databases: List[str]
    download_all: bool = False
    custom_amount: int = 50
    email: Optional[str] = None
    password: Optional[str] = None
    show_browser: bool = False

class CleaningRequest(BaseModel):
    output_name: str
    csv_files: Optional[Dict[str, str]] = None

class AnalysisRequest(BaseModel):
    output_name: str
    csv_file: Optional[str] = None

# Crear aplicación FastAPI
app = FastAPI(title="Academic Analysis API")
api = AcademicAnalysisAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="."), name="static")

# === Endpoints básicos ===
@app.get("/")
async def read_root():
    """Página principal - interfaz HTML"""
    return FileResponse("interface.html")

@app.get("/status")
async def get_status():
    """Obtener estado actual del sistema"""
    return api.get_status()

# === Endpoints de autenticación ===
@app.post("/check-cookies")
async def check_cookies(databases: List[str]):
    """Verificar cookies de bases de datos"""
    return api.check_cookies(databases)

@app.post("/login")
async def login_databases(request: LoginRequest):
    """Login en bases de datos seleccionadas"""
    return api.login_databases(
        request.databases,
        request.email,
        request.password,
        request.show_browser
    )

# === Endpoints de scraping ===
@app.post("/get-availability")
async def get_availability(request: ScrapingRequest):
    """Obtener disponibilidad de artículos"""
    return api.get_availability(
        request.query,
        request.databases,
        request.email,
        request.password,
        request.show_browser
    )

@app.post("/scrape")
async def start_scraping(request: ScrapingRequest):
    """Iniciar proceso de scraping"""
    return api.start_scraping(
        request.query,
        request.databases,
        request.download_all,
        request.custom_amount,
        request.email,
        request.password,
        request.show_browser
    )

# === Endpoints de limpieza ===
@app.post("/clean")
async def start_cleaning(request: CleaningRequest):
    """Iniciar limpieza de datos"""
    return api.start_cleaning(request.output_name, request.csv_files)

# === Endpoints de análisis ===
@app.post("/analyze")
async def start_analysis(request: AnalysisRequest):
    """Iniciar análisis de datos"""
    return api.start_analysis(request.output_name, request.csv_file)

@app.post("/analyze-concepts")
async def analyze_concepts(csv_file: Optional[str] = None, top_k: int = 15):
    """Análisis de conceptos GAIE"""
    return api.analyze_concepts(csv_file, top_k)

@app.post("/analyze-clustering")
async def analyze_clustering(
    csv_file: Optional[str] = None,
    max_docs: int = 150,
    algorithms: Optional[List[str]] = None
):
    """Análisis de clustering jerárquico"""
    return api.analyze_hierarchical_clustering(csv_file, max_docs, algorithms)

@app.post("/analyze-citation-graph")
async def analyze_citation_graph(csv_file: Optional[str] = None):
    """Análisis de grafo de citaciones"""
    return api.analyze_citation_graph(csv_file)

@app.post("/analyze-cooccurrence")
async def analyze_cooccurrence(
    csv_file: Optional[str] = None,
    top_k_terms: int = 40,
    min_cooc: int = 2
):
    """Análisis de coocurrencia de términos"""
    return api.analyze_term_cooccurrence_graph(csv_file, top_k_terms, min_cooc)

# === Endpoints de pipeline completo ===
@app.post("/pipeline")
async def start_pipeline(request: ScrapingRequest):
    """Ejecutar pipeline completo"""
    return api.start_full_pipeline(
        query=request.query,
        databases=request.databases,
        download_all=request.download_all,
        custom_amount=request.custom_amount,
        output_name="pipeline_output",
        email=request.email,
        password=request.password,
        show_browser=request.show_browser
    )