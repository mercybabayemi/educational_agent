"""FastAPI application entry point."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes.health import router as health_router
from app.api.routes.skills import router as skills_router
from app.api.routes.curriculum import router as curriculum_router
from app.api.routes.content import router as content_router
from app.api.routes.assessment import router as assessment_router
from app.api.routes.insights import router as insights_router
from app.api.routes.ingest import router as ingest_router

RAW_DATA_DIR = Path("data/raw")
_REAL_FILE_SUFFIXES = {".pdf", ".docx", ".doc", ".txt"}


def _real_files_exist() -> bool:
    if not RAW_DATA_DIR.exists():
        return False
    return any(
        f.suffix.lower() in _REAL_FILE_SUFFIXES
        for f in RAW_DATA_DIR.rglob("*") if f.is_file()
    )


async def _bootstrap_rag() -> None:
    """Populate vectorstore before the app starts accepting requests.

    Priority:
    1. Real curriculum files in data/raw/ → ingest_files()
    2. No real files → ingest_synthetic() as fallback
    Errors are caught so a failed bootstrap never prevents the app from starting.
    """
    from app.rag.vectorstore import get_document_count

    try:
        count = get_document_count()
    except Exception as exc:
        logger.warning(f"Could not read vectorstore on startup: {exc}")
        return

    if count > 0:
        logger.info(f"Vectorstore ready — {count} chunks already loaded. Skipping bootstrap.")
        return

    logger.info("Vectorstore is empty — running startup RAG bootstrap (app will accept requests after this).")

    try:
        if _real_files_exist():
            logger.info(f"Found curriculum files in {RAW_DATA_DIR} — ingesting real documents.")
            from app.rag.ingestion.ingest import ingest_files
            await asyncio.to_thread(ingest_files, RAW_DATA_DIR)
        else:
            logger.info("No curriculum files found — generating synthetic bootstrap content.")
            from app.rag.ingestion.ingest import ingest_synthetic
            await asyncio.to_thread(ingest_synthetic)

        final_count = get_document_count()
        logger.info(f"Startup bootstrap complete — {final_count} chunks in vectorstore. Ready.")
    except Exception as exc:
        logger.error(
            f"Startup RAG bootstrap failed: {exc}. "
            "App will start but RAG context will be empty. Re-trigger via Gradio UI."
        )


async def _warmup_models() -> None:
    """Eagerly load both LLM pipelines so they're cached before any request."""
    from app.core.llm import get_phi_pipeline, get_judge_pipeline

    try:
        await asyncio.to_thread(get_phi_pipeline)
        logger.info("Primary model (Phi-3) warmed up.")
    except Exception as exc:
        logger.warning(f"Primary model warmup failed: {exc}")

    try:
        await asyncio.to_thread(get_judge_pipeline)
        logger.info("Judge model (TinyLlama) warmed up.")
    except Exception as exc:
        logger.warning(f"Judge model warmup failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bootstrap RAG then pre-load both LLM pipelines before accepting requests."""
    await _bootstrap_rag()
    await _warmup_models()
    yield


app = FastAPI(
    title="AfriPed",
    description=(
        "African Pedagogical Evaluation Framework — AI-powered educational content generation "
        "for Nigerian and West African learners. Generates curriculum-aligned, culturally authentic "
        "content across four pillars: Curriculum, Content, Assessment, and Insights."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREFIX = "/api/v1"
app.include_router(health_router,     prefix=PREFIX, tags=["Health"])
app.include_router(skills_router,     prefix=PREFIX, tags=["Skill Library"])
app.include_router(curriculum_router, prefix=PREFIX, tags=["Curriculum"])
app.include_router(content_router,    prefix=PREFIX, tags=["Content"])
app.include_router(assessment_router, prefix=PREFIX, tags=["Assessment"])
app.include_router(insights_router,   prefix=PREFIX, tags=["Insights"])
app.include_router(ingest_router,     prefix=PREFIX, tags=["Ingestion"])
