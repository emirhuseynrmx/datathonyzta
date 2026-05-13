import logging
import sys
from types import FrameType

from loguru import logger

from src.config import settings


class InterceptHandler(logging.Handler):
    """Standart logging mesajlarını yakalayıp Loguru'ya yönlendiren köprü."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_custom_logger() -> None:
    """Üretim seviyesi loglama: Hem renkli terminal hem de MLOps uyumlu JSONL dosya kaydı."""
    logger.remove()

    # 1. Terminal (İnsan dostu)
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        level="INFO"
    )

    # 2. Dosya (Makine dostu - JSONL)
    log_file = settings.PROJECT_DIR / "logs/yzta_pipeline.jsonl"
    logger.add(
        log_file,
        serialize=True,  # Logları JSON formatında tutar (MLOps standardı)
        rotation="100 MB",
        retention="10 days"
    )

    # Dış kütüphaneleri ele geçir
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for _name in ["lightgbm", "optuna", "xgboost"]:
        _logger = logging.getLogger(_name)
        _logger.handlers = [InterceptHandler()]
        _logger.propagate = False


setup_custom_logger()
