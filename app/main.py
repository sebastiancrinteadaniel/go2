from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings

app = FastAPI(title="Go2 Dashboard")

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
