from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from academic_analysis_gui import AcademicAnalysisAPI  # Importar la clase API

app = FastAPI(title="API del sistema de análisis académico")
api = AcademicAnalysisAPI()  # Instanciar la API

# --- Middleware CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Ruta principal ---
@app.get("/", include_in_schema=False)
async def root():
    file_path = os.path.join(os.path.dirname(__file__), "interface.html")
    return FileResponse(file_path)

# --- Archivos estáticos ---
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# --- Endpoints que replican la funcionalidad de main.py ---
@app.post("/analyze")
async def analyze_data():
    result = api.analyze()  # Método de tu clase AcademicAnalysisAPI
    return result

@app.get("/get_results")
async def get_results():
    return api.get_results()  # Método para obtener resultados