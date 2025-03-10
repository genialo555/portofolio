from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
import time
import logging
import uvicorn
import os
from typing import Dict, Any, List, Optional

from .routes import models_router, health_router
from .routes.rag import router as rag_router
from .routes.kag import router as kag_router
from .routes.hybrid import router as hybrid_router
from .routes.agents import router as agents_router
from .routes.orchestration import router as orchestration_router
from .middleware import LoggingMiddleware, TimingMiddleware

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ml_api")

# Création de l'application FastAPI
app = FastAPI(
    title="ML Service API",
    description="API pour les services de machine learning RAG, KAG, Hybride et Agents",
    version="1.0.0",
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middlewares personnalisés
app.add_middleware(TimingMiddleware)
app.add_middleware(LoggingMiddleware)

# Attacher les routeurs
app.include_router(models_router, prefix="/api/v1/models", tags=["Models"])
app.include_router(health_router, prefix="/api/v1/health", tags=["Health"])
app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(kag_router, prefix="/api/v1/kag", tags=["KAG"])
app.include_router(hybrid_router, prefix="/api/v1/hybrid", tags=["Hybrid"])
app.include_router(agents_router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(orchestration_router, prefix="/api/v1/orchestration", tags=["Orchestration"])

# Gestionnaire d'exception global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Exception non gérée: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Une erreur interne est survenue", "detail": str(exc)},
    )

# Route racine pour rediriger vers la documentation
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "API ML Service - Accédez à /docs pour la documentation"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 