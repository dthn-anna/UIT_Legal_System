from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.container import ServiceContainer


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    container = ServiceContainer(settings)
    await container.startup()
    app.state.container = container

    try:
        yield
    finally:
        await container.shutdown()


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    router,
    prefix=settings.api_prefix,
)

if FRONTEND_DIR.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="frontend",
    )


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "status": "running",
        "chat": "/chat",
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }


@app.get("/chat", include_in_schema=False)
async def chat_page():
    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():
        return {
            "detail": (
                "Frontend not found. "
                "Place index.html, style.css and app.js in the frontend folder."
            )
        }

    return FileResponse(index_file)
