from typing import Dict, Any, List, Optional
import asyncio
import httpx
from pathlib import Path
import base64
import io
from PIL import Image
import torch

class ImageTeacherModel:
    """Modèle enseignant pour la génération d'images utilisant R1 (SDXL)."""
    
    def __init__(self):
        self.r1_api_key = os.getenv('REPLICATE_API_KEY')
        self.model_versions = {
            'sdxl': 'stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b',
            'sdxl_turbo': 'stability-ai/sdxl-turbo:1d9b2f4aad64e55feafaa49aa50b4542933ac157b1d99c8c8a46e1f97e46b5df',
            'pixart': 'pixart-alpha/PixArt-XL-2-1024-MS:1d1c4c6e6d31c32cfeb8f9c3eba0645aa23c18a87f8757e0ae8eaf657a783c18'
        }
        
        # Configuration pour la latence minimale
        self.default_config = {
            'num_inference_steps': 20,  # Réduit pour la vitesse
            'guidance_scale': 7.5,
            'refiner_inference_steps': 10,
            'width': 1024,
            'height': 1024
        }
        
    async def generate_image(self, 
                           prompt: str,
                           negative_prompt: Optional[str] = None,
                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Génère une image avec feedback d'enseignant."""
        
        # Utilise SDXL Turbo pour la première passe rapide
        initial_result = await self._generate_with_r1(
            'sdxl_turbo',
            prompt,
            negative_prompt,
            {'num_inference_steps': 1}  # Ultra rapide
        )
        
        # Analyse la qualité avec PixArt
        quality_feedback = await self._analyze_quality(initial_result['image'])
        
        if quality_feedback['score'] < 0.7:
            # Si la qualité n'est pas suffisante, utilise SDXL complet
            final_result = await self._generate_with_r1(
                'sdxl',
                prompt,
                negative_prompt,
                self.default_config
            )
        else:
            final_result = initial_result
            
        return {
            'image': final_result['image'],
            'feedback': quality_feedback,
            'model_used': 'sdxl_turbo' if quality_feedback['score'] >= 0.7 else 'sdxl',
            'generation_time': final_result['generation_time']
        }
        
    async def _generate_with_r1(self,
                               model_key: str,
                               prompt: str,
                               negative_prompt: Optional[str],
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une image avec R1."""
        async with httpx.AsyncClient() as client:
            start_time = asyncio.get_event_loop().time()
            
            response = await client.post(
                'https://api.replicate.com/v1/predictions',
                headers={'Authorization': f"Token {self.r1_api_key}"},
                json={
                    'version': self.model_versions[model_key],
                    'input': {
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        **config
                    }
                }
            )
            
            prediction = response.json()
            
            # Attente du résultat avec timeout
            while prediction['status'] not in ['succeeded', 'failed']:
                await asyncio.sleep(0.5)
                response = await client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction['id']}",
                    headers={'Authorization': f"Token {self.r1_api_key}"}
                )
                prediction = response.json()
                
            generation_time = asyncio.get_event_loop().time() - start_time
            
            return {
                'image': prediction['output'][0],
                'generation_time': generation_time
            }
            
    async def _analyze_quality(self, image_url: str) -> Dict[str, float]:
        """Analyse la qualité de l'image avec PixArt."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                'https://api.replicate.com/v1/predictions',
                headers={'Authorization': f"Token {self.r1_api_key}"},
                json={
                    'version': self.model_versions['pixart'],
                    'input': {
                        'image': image_url,
                        'task': 'quality_assessment'
                    }
                }
            )
            
            result = response.json()
            return {
                'score': float(result['output']['quality_score']),
                'aspects': result['output']['quality_aspects']
            } 