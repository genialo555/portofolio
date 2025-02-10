from typing import Dict, Any, Union, List
import asyncio
from pathlib import Path
import aiofiles
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import torch
from transformers import AutoProcessor, AutoModel

class DocumentConverterAgent:
    """Agent spécialisé dans la conversion de documents multi-formats."""
    
    def __init__(self):
        self.image_processor = AutoProcessor.from_pretrained("microsoft/resnet-50")
        self.vision_model = AutoModel.from_pretrained("microsoft/resnet-50")
        self.anomaly_detector = AnomalyDetector("conversion_monitoring")
        
    async def convert_document(self, 
                             source: Union[str, Path, bytes],
                             source_type: str) -> Dict[str, Any]:
        """Convertit différents types de documents en format unifié."""
        
        # Vérification d'anomalie
        conversion_check = self.anomaly_detector.check({
            "source_type": source_type,
            "source_size": len(source) if isinstance(source, bytes) else Path(source).stat().st_size
        })
        
        if conversion_check.has_anomaly:
            return {"error": conversion_check.details}
            
        try:
            if source_type == "image":
                return await self.process_image(source)
            elif source_type == "pdf":
                return await self.process_pdf(source)
            elif source_type == "video":
                return await self.process_video(source)
            else:
                return await self.process_text(source)
                
        except Exception as e:
            self.anomaly_detector.report_error(str(e))
            return {"error": f"Conversion failed: {str(e)}"}
            
    async def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Traite les images avec OCR et analyse visuelle."""
        image = Image.open(image_path)
        
        # OCR pour le texte
        text = pytesseract.image_to_string(image)
        
        # Analyse visuelle
        inputs = self.image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            visual_features = self.vision_model(**inputs).last_hidden_state
            
        return {
            "text": text,
            "visual_features": visual_features,
            "metadata": {
                "size": image.size,
                "mode": image.mode
            }
        } 