@app.get("/", response_class=HTMLResponse)
def index():
    ui_path = "interface.html"
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h3>Academic Analysis System API</h3><p>Use endpoints to run analysis.</p>", status_code=200)
