from typing import Dict, Any, Optional
import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv

class FreeModelGateway:
    """Passerelle vers les services d'API gratuits."""
    
    def __init__(self):
        load_dotenv()
        self.api_keys = {
            'huggingface': os.getenv('HF_API_KEY'),  # Gratuit
            'replicate': os.getenv('REPLICATE_API_KEY'),  # R1 - Crédits gratuits
            'together': os.getenv('TOGETHER_API_KEY'),  # Together.ai - Tier gratuit
            'ollama': 'local',  # Local gratuit
            'gradio': None  # Endpoints Gradio publics
        }
        
        # URLs des endpoints gratuits
        self.endpoints = {
            'huggingface': 'https://api-inference.huggingface.co/models',
            'replicate': 'https://api.replicate.com/v1',
            'together': 'https://api.together.xyz/inference',
            'ollama': 'http://localhost:11434',
            'gradio': 'https://huggingface.co/spaces'
        }
        
        # Modèles R1 populaires et gratuits
        self.r1_models = {
            'llama': 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
            'sdxl': 'stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b',
            'mistral': 'mistralai/mistral-7b-instruct-v0.1'
        }
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def query_free_model(self, 
                             provider: str,
                             model: str,
                             inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Interroge un modèle via une API gratuite."""
        try:
            if provider == 'replicate':
                return await self._query_replicate(model, inputs)
            elif provider == 'huggingface':
                return await self._query_huggingface(model, inputs)
            elif provider == 'together':
                return await self._query_together(model, inputs)
            elif provider == 'ollama':
                return await self._query_ollama(model, inputs)
            elif provider == 'gradio':
                return await self._query_gradio(model, inputs)
        except Exception as e:
            # Fallback vers un autre provider si erreur
            return await self._fallback_query(provider, model, inputs, str(e))
            
    async def _query_replicate(self, model: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Interroge l'API Replicate (R1)."""
        async with httpx.AsyncClient() as client:
            # Utilise le modèle R1 si spécifié, sinon utilise l'ID fourni
            model_id = self.r1_models.get(model, model)
            
            # Création de la prédiction
            response = await client.post(
                f"{self.endpoints['replicate']}/predictions",
                headers={'Authorization': f"Token {self.api_keys['replicate']}"},
                json={
                    'version': model_id,
                    'input': inputs
                }
            )
            prediction = response.json()
            
            # Attente du résultat
            while prediction['status'] not in ['succeeded', 'failed']:
                await asyncio.sleep(1)
                response = await client.get(
                    f"{self.endpoints['replicate']}/predictions/{prediction['id']}",
                    headers={'Authorization': f"Token {self.api_keys['replicate']}"}
                )
                prediction = response.json()
                
            return prediction
            
    async def _query_together(self, model: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Interroge l'API Together.ai (alternative gratuite)."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoints['together']}/v1/completions",
                headers={'Authorization': f"Bearer {self.api_keys['together']}"},
                json={
                    'model': model,
                    'prompt': inputs['prompt'],
                    'max_tokens': inputs.get('max_tokens', 512)
                }
            )
            return response.json()
            
    async def _query_ollama(self, model: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Interroge un modèle Ollama local (gratuit)."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoints['ollama']}/api/generate",
                json={
                    'model': model,  # llama2, mistral, etc.
                    'prompt': inputs['prompt'],
                    'stream': False
                }
            )
            return response.json()
            
    async def _query_huggingface(self, model: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Interroge l'API HuggingFace (tier gratuit)."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoints['huggingface']}/{model}",
                headers={'Authorization': f"Bearer {self.api_keys['huggingface']}"},
                json=inputs
            )
            return response.json() 