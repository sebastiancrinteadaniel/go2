import sys
import os

# Ensure the root directory is in sys.path so 'app.*' imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router, close_all

from app.core.config import settings
from app.core.logger import setup_logger
import logging
from contextlib import asynccontextmanager

setup_logger()


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA available — device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("PyTorch CUDA not available — running on CPU")
    except ImportError:
        logger.warning("PyTorch not installed.")

    yield

    await close_all()
    logger.info("Active WebRTC session closed on shutdown.")


app = FastAPI(title="Go2 Dashboard", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    import socket

    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(("10.255.255.255", 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = "127.0.0.1"
        finally:
            s.close()
        return IP

    local_ip = get_ip()
    logger.info("\n" + "=" * 50)
    logger.info("🚀 Dashboard is running!")
    logger.info("👉 Local:   http://localhost:8000")
    logger.info(f"👉 Network: http://{local_ip}:8000")
    logger.info("=" * 50 + "\n")

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
