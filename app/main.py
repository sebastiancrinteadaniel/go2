import sys
import os

# Ensure the root directory is in sys.path so 'app.*' imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

from app.core.config import settings
from app.core.logger import setup_logger
import logging
from contextlib import asynccontextmanager

setup_logger()


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        import psutil
        import os

        # Determine network interface - eth0 for Jetson or lo for local/simulation
        is_jetson = os.path.exists("/etc/nv_tegra_release")
        has_eth0 = "eth0" in psutil.net_if_addrs()

        if is_jetson and has_eth0:
            ChannelFactoryInitialize(0, "eth0")
            logger.info("Initialized Unitree SDK with eth0")
        else:
            ChannelFactoryInitialize(0, "lo")
            logger.info("Initialized Unitree SDK with loopback (lo)")
    except ImportError:
        logger.warning("'unitree_sdk2py' not found. SDK capabilities disabled.")
    except Exception as e:
        logger.error(f"Error initializing Unitree SDK: {e}")

    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA available — device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("PyTorch CUDA not available — running on CPU")
    except ImportError:
        logger.warning("PyTorch not installed.")

    yield


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
