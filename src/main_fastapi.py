# main_fastapi.py
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import uvicorn
import os
import tempfile
import traceback

# Importar tus módulos (ajusta rutas de import si están en src/ u otro paquete)
# Ejemplo: from src.AcademicSortingAnalyzer import AcademicSortingAnalyzer
# Ajusta los imports según tu estructura de carpetas.
try:
    from AcademicSortingAnalyzer import analyze_academic_data, AcademicSortingAnalyzer
except Exception:
    # Si está en carpeta src:
    try:
        from src.AcademicSortingAnalyzer import analyze_academic_data, AcademicSortingAnalyzer
    except Exception:
        AcademicSortingAnalyzer = None
        analyze_academic_data = None

try:
    from SimilitudTextualIA import SimilitudTextualIA
except Exception:
    SimilitudTextualIA = None

try:
    from BibliometricVisualizer import BibliometricVisualizer
except Exception:
    BibliometricVisualizer = None

try:
    from CitationNetworkAnalyzer import CitationNetworkAnalyzer
except Exception:
    CitationNetworkAnalyzer = None

app = FastAPI(title="AAS API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def index():
    # Si deseas servir tu interface.html directamente
    ui_path = "interface.html"
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h3>Academic Analysis System API</h3><p>Use endpoints to run analysis.</p>", status_code=200)

# Endpoint para ejecutar el scraping / ingesta: recibe un archivo RIS/BibTeX o usa fuentes internas
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        out_dir = "uploaded"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, file.filename)
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
        return {"filename": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para lanzar el análisis completo (wrapper)
@app.post("/analyze")
def analyze(background_tasks: BackgroundTasks, dataset_path: str = None, output_base: str = "aas_output"):
    """
    Inicia el pipeline completo en background y retorna un ID o ruta.
    dataset_path: ruta al CSV/parquet que se desea analizar (opcional)
    """
    try:
        if analyze_academic_data is None and AcademicSortingAnalyzer is None:
            return JSONResponse(status_code=500, content={"error": "AcademicSortingAnalyzer not available in container."})

        # Si se pasó un dataset local, usarlo; si no, buscar data_master.parquet
        if dataset_path and os.path.exists(dataset_path):
            csv_path = dataset_path
            analyzer = AcademicSortingAnalyzer(csv_path)
            background_tasks.add_task(analyzer.generate_complete_report, output_base)
            return {"status": "started", "output_base": output_base}
        else:
            # fallback: if analyze_academic_data wrapper exists
            if analyze_academic_data:
                background_tasks.add_task(analyze_academic_data, "data_master.parquet", output_base)
                return {"status": "started", "output_base": output_base}
            else:
                return JSONResponse(status_code=400, content={"error": "No dataset provided and no wrapper available"})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para pedir resultados (por ejemplo un PNG generado)
@app.get("/download/{filename}")
def download_file(filename: str):
    path = os.path.join(".", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

# Endpoint ejemplo para similitud IA (pairwise)
@app.post("/similarity/ia")
def similarity_ia(text1: str, text2: str):
    if SimilitudTextualIA is None:
        raise HTTPException(status_code=500, detail="SimilitudTextualIA module not available.")
    model = SimilitudTextualIA()
    score = model.similitud_semantica(text1, text2)  # adapta nombre método
    return {"score": float(score)}

# Endpoint para visualizaciones
@app.post("/visualize")
def visualize(output_base: str = "viz_output"):
    if BibliometricVisualizer is None:
        raise HTTPException(status_code=500, detail="BibliometricVisualizer not available.")
    viz = BibliometricVisualizer()
    files = viz.generate_all_plots(output_base)  # adapta según firma real
    return {"files": files}

# Si ejecutas directamente
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main_fastapi:app", host="0.0.0.0", port=port, log_level="info")
