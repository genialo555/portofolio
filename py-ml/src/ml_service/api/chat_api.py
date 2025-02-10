from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import asyncio
from ..pilpoul.debate_engine import PilpoulEngine
from ..agents.instagram.influencer_agent import InstagramInfluencerAgent
from pydantic import BaseModel

class ChatMessage(BaseModel):
    message: str
    context: Dict[str, Any] = {}
    session_id: str

app = FastAPI(title="Pilpoul Chatbot API")

# Configuration CORS pour l'iframe
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatManager:
    def __init__(self):
        self.pilpoul = PilpoulEngine()
        self.instagram_agent = InstagramInfluencerAgent(niche='general')
        self.active_sessions: Dict[str, Dict] = {}

    async def process_message(self, msg: ChatMessage) -> Dict[str, Any]:
        """Traite un message et retourne la réponse."""
        if msg.session_id not in self.active_sessions:
            self.active_sessions[msg.session_id] = {
                'history': [],
                'context': msg.context
            }

        # Analyse du contexte pour choisir l'agent
        if 'instagram' in msg.message.lower():
            response = await self.instagram_agent.analyze_profile({
                'query': msg.message,
                **msg.context
            })
        else:
            # Utilise le moteur pilpoul par défaut
            response = await self.pilpoul.analyze_instagram_strategy({
                'query': msg.message,
                'history': self.active_sessions[msg.session_id]['history']
            })

        # Mise à jour de l'historique
        self.active_sessions[msg.session_id]['history'].append({
            'user': msg.message,
            'bot': response
        })

        return {
            'response': response,
            'session_id': msg.session_id,
            'context': self.active_sessions[msg.session_id]['context']
        }

chat_manager = ChatManager()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            msg = ChatMessage(**data)
            response = await chat_manager.process_message(msg)
            await websocket.send_json(response)
    except Exception as e:
        await websocket.close(code=1000)

@app.post("/api/chat")
async def chat_endpoint(msg: ChatMessage):
    """Endpoint HTTP pour le chat."""
    return await chat_manager.process_message(msg) 