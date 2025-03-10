import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

logger = logging.getLogger("ml_api.middleware")

class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware pour mesurer le temps d'exécution des requêtes."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Ajout du temps d'exécution dans les en-têtes de réponse
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log si la requête prend trop de temps (plus de 1 seconde)
        if process_time > 1:
            logger.warning(f"Requête lente: {request.url.path} a pris {process_time:.2f}s")
        
        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour logger les requêtes et les réponses."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Log de la requête entrante
        logger.info(f"Requête entrante: {request.method} {request.url.path}")
        
        # Récupération de l'adresse IP du client
        forwarded_for = request.headers.get("X-Forwarded-For")
        client_ip = forwarded_for.split(",")[0] if forwarded_for else request.client.host
        
        # Log des informations supplémentaires en debug
        logger.debug(f"Client IP: {client_ip}, User-Agent: {request.headers.get('User-Agent')}")
        
        try:
            # Traitement de la requête
            response = await call_next(request)
            
            # Log de la réponse
            logger.info(f"Réponse: {request.method} {request.url.path} - Status: {response.status_code}")
            
            return response
        except Exception as e:
            # Log des exceptions
            logger.error(f"Exception dans le traitement de {request.url.path}: {str(e)}", exc_info=True)
            raise 