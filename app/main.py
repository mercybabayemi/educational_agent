"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.skills import router as skills_router
from app.api.routes.curriculum import router as curriculum_router
from app.api.routes.content import router as content_router
from app.api.routes.assessment import router as assessment_router
from app.api.routes.insights import router as insights_router

app = FastAPI(
    title="EnumVerse AI",
    description=(
        "AI-powered educational content generation for Nigerian and West African learners. "
        "Generates curriculum-aligned, culturally authentic content across four pillars: "
        "Curriculum, Content, Assessment, and Insights."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
