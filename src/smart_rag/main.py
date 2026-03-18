from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.v1 import router as v1_router
from .config import get_settings
from .logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["system"])
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    app.include_router(v1_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
