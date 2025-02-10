from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio

@dataclass
class InstagramMetrics:
    engagement_rate: float
    follower_growth: float
    content_performance: Dict[str, float]
    audience_insights: Dict[str, Any]

@dataclass
class ContentStrategy:
    post_types: List[str]
    optimal_timing: Dict[str, List[str]]
    hashtag_groups: Dict[str, List[str]]
    caption_templates: List[str]

class InstagramInfluencerAgent:
    """Agent spécialisé pour l'assistance des influenceurs Instagram."""
    
    def __init__(self, niche: str):
        self.niche = niche
        self.base_model = 'Qwen/Qwen2.5-7B-Chat'  # Excellent pour le marketing
        self.strategies = {
            'beauty': ContentStrategy(
                post_types=['Reels', 'Carousel', 'Stories'],
                optimal_timing={'weekday': ['18:00', '21:00'], 'weekend': ['12:00', '15:00']},
                hashtag_groups={
                    'makeup': ['#beautytips', '#makeuptutorial'],
                    'skincare': ['#skincareregime', '#glowingskin']
                }
            ),
            'tech': ContentStrategy(
                post_types=['Tutorial', 'Review', 'News'],
                optimal_timing={'weekday': ['10:00', '16:00'], 'weekend': ['14:00']},
                hashtag_groups={
                    'gadgets': ['#techreview', '#newtech'],
                    'coding': ['#codinglife', '#developerlife']
                }
            )
            # ... autres niches
        }
        
    async def analyze_profile(self, profile_data: Dict) -> InstagramMetrics:
        """Analyse approfondie du profil Instagram."""
        prompt = f"""
        Analysez ce profil Instagram de {self.niche} :
        
        Statistiques : {profile_data['stats']}
        Contenu récent : {profile_data['recent_posts']}
        Audience : {profile_data['audience_data']}
        
        Fournissez une analyse détaillée pour :
        1. Taux d'engagement
        2. Croissance des abonnés
        3. Performance du contenu
        4. Insights sur l'audience
        """
        
        analysis = await self._query_model(prompt)
        return self._parse_metrics(analysis)
        
    async def generate_content_plan(self, metrics: InstagramMetrics) -> Dict[str, Any]:
        """Génère un plan de contenu personnalisé."""
        strategy = self.strategies.get(self.niche, self.strategies['general'])
        
        prompt = f"""
        Créez un plan de contenu Instagram pour un influenceur {self.niche} :
        
        Métriques actuelles :
        {metrics}
        
        Stratégie de base :
        {strategy}
        
        Générez un plan incluant :
        1. Calendrier de publication
        2. Thèmes de contenu
        3. Formats recommandés
        4. Hashtags optimisés
        5. Suggestions d'engagement
        """
        
        return await self._query_model(prompt)
        
    async def optimize_monetization(self, profile_metrics: InstagramMetrics) -> Dict[str, Any]:
        """Optimise la stratégie de monétisation."""
        prompt = f"""
        Développez une stratégie de monétisation pour un influenceur {self.niche} :
        
        Métriques actuelles :
        {profile_metrics}
        
        Proposez :
        1. Opportunités de partenariats
        2. Produits/services potentiels
        3. Stratégie de pricing
        4. Canaux de revenus
        5. Plan d'action
        """
        
        return await self._query_model(prompt) 