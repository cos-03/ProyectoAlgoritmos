# main_fastapi.py
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import uvicorn
import os
import traceback
from typing import Optional, Dict, Any

# Importar módulos desde src/ (estructura del repo: src/*.py)
try:
    from src.AcademicSortingAnalyzer import analyze_academic_data, AcademicSortingAnalyzer
except Exception:
    analyze_academic_data = None
    AcademicSortingAnalyzer = None

try:
    from src.SimilitudTextualIA import SimilitudTextualIA
except Exception:
    SimilitudTextualIA = None

try:
    from src.BibliometricVisualizer import BibliometricVisualizer
except Exception:
    BibliometricVisualizer = None

try:
    from src.CitationNetworkAnalyzer import CitationNetworkAnalyzer
except Exception:
    CitationNetworkAnalyzer = None

app = FastAPI(title="AAS API", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    ui_path = "interface.html"
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h3>Academic Analysis System API</h3><p>Use endpoints to run analysis.</p>", status_code=200)


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, str]:
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


@app.post("/analyze")
def analyze(background_tasks: BackgroundTasks, dataset_path: Optional[str] = None, output_base: str = "aas_output"):
    """
    Inicia el pipeline completo en background y retorna inmediatamente.
    Si dataset_path se proporciona y existe, lo usa; sino intenta usar analyze_academic_data()
    con 'data_master.parquet' si está disponible.
    """
    try:
        if AcademicSortingAnalyzer is None and analyze_academic_data is None:
            raise HTTPException(status_code=500, detail="AcademicSortingAnalyzer not available on server.")

        if dataset_path:
            if not os.path.exists(dataset_path):
                raise HTTPException(status_code=400, detail=f"Dataset not found: {dataset_path}")
            # lanzar análisis con clase
            if AcademicSortingAnalyzer is None:
                raise HTTPException(status_code=500, detail="AcademicSortingAnalyzer class not available.")
            analyzer = AcademicSortingAnalyzer(dataset_path)
            background_tasks.add_task(analyzer.generate_complete_report, output_base)
            return {"status": "started", "mode": "class", "output_base": output_base}
        else:
            # fallback wrapper
            if analyze_academic_data:
                background_tasks.add_task(analyze_academic_data, "data_master.parquet", output_base)
                return {"status": "started", "mode": "wrapper", "output_base": output_base}
            else:
                raise HTTPException(status_code=400, detail="No dataset provided and no wrapper available")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
def download_file(filename: str):
    path = os.path.join(".", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/similarity/ia")
async def similarity_ia(payload: Dict[str, str]):
    """
    Espera JSON: { "text1": "...", "text2": "..." }
    """
    try:
        text1 = payload.get("text1", "")
        text2 = payload.get("text2", "")
        if SimilitudTextualIA is None:
            raise HTTPException(status_code=500, detail="SimilitudTextualIA module not available.")
        model = SimilitudTextualIA()
        # Buscamos nombres de método comunes
        if hasattr(model, "similitud_semantica"):
            score = model.similitud_semantica(text1, text2)
        elif hasattr(model, "similarity") :
            score = model.similarity(text1, text2)
        elif hasattr(model, "compute_similarity"):
            score = model.compute_similarity(text1, text2)
        else:
            raise HTTPException(status_code=500, detail="No known similarity method found in SimilitudTextualIA.")
        return {"score": float(score)}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize")
def visualize(output_base: str = "viz_output"):
    try:
        if BibliometricVisualizer is None:
            raise HTTPException(status_code=500, detail="BibliometricVisualizer not available.")
        viz = BibliometricVisualizer()
        # intentar métodos comunes (gama de nombres)
        for candidate in ("generate_all_plots", "generate_visualizations", "run_all_visualizations", "create_visualizations"):
            if hasattr(viz, candidate):
                gen = getattr(viz, candidate)
                files = gen(output_base)
                return {"files": files}
        # fallback: si la clase implementa 'run' o '__call__'
        if hasattr(viz, "run"):
            files = viz.run(output_base)
            return {"files": files}
        raise HTTPException(status_code=500, detail="No visualization method found in BibliometricVisualizer.")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Ejecutar por nombre de módulo funciona; si prefieres, usar uvicorn.run(app, ...)
    uvicorn.run("main_fastapi:app", host="0.0.0.0", port=port, log_level="info")
