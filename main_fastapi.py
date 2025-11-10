from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

app = FastAPI(title="API del sistema de análisis académico")

@app.get("/", response_class=HTMLResponse)
def index():
    # Buscar archivo HTML en posibles ubicaciones
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base_dir, "interface.html"),
        os.path.join(base_dir, "src", "interface.html"),
        os.path.join(base_dir, "public", "index.html"),
    ]

    for path in paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)

    # fallback (si no existe el archivo)
    return HTMLResponse(
        content="""
        <h2>API del sistema de análisis académico</h2>
        <p>Utilice los endpoints para ejecutar el análisis.</p>
        """,
        status_code=200,
    )
