from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

from app.core.config import settings
from app.core.logger import setup_logger

setup_logger()

import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the Unitree SDK ChannelFactory globally before any subscribers
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
        
    yield

app = FastAPI(title="Go2 Dashboard", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
