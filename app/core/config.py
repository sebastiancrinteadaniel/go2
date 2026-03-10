import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    STATIC_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
    )

    CAMERA_WIDTH: int = 1920
    CAMERA_HEIGHT: int = 1080
    CAMERA_FPS: int = 30


settings = Settings()

os.makedirs(settings.STATIC_DIR, exist_ok=True)
