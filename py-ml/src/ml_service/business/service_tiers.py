from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class ServiceTier(Enum):
    STARTER = "starter"
    GROWTH = "growth"
    ELITE = "elite"

@dataclass
class PricingTier:
    name: ServiceTier
    price: float
    features: List[str]
    api_calls: int
    support_level: str

class InstagramMLService:
    """Service ML pour influenceurs Instagram."""
    
    def __init__(self):
        self.pricing = {
            ServiceTier.STARTER: PricingTier(
                name=ServiceTier.STARTER,
                price=49.99,
                features=[
                    "Analyse basique du profil",
                    "Suggestions de hashtags",
                    "Planning hebdomadaire",
                    "5 niches disponibles"
                ],
                api_calls=100,
                support_level="Email"
            ),
            ServiceTier.GROWTH: PricingTier(
                name=ServiceTier.GROWTH,
                price=149.99,
                features=[
                    "Analyse avancée du profil",
                    "Stratégie de contenu personnalisée",
                    "Optimisation monétisation",
                    "Toutes les niches",
                    "API Instagram intégrée",
                    "Analytics en temps réel"
                ],
                api_calls=500,
                support_level="Priority + Chat"
            ),
            ServiceTier.ELITE: PricingTier(
                name=ServiceTier.ELITE,
                price=499.99,
                features=[
                    "IA personnalisée dédiée",
                    "Fine-tuning sur votre profil",
                    "Stratégie multi-plateformes",
                    "Prédictions de tendances",
                    "Accès API illimité",
                    "Manager de compte dédié"
                ],
                api_calls=float('inf'),
                support_level="24/7 Dedicated"
            )
        }
        
    async def create_subscription(self, tier: ServiceTier, user_data: Dict) -> Dict:
        """Crée un nouvel abonnement."""
        tier_config = self.pricing[tier]
        
        return {
            'subscription_id': str(uuid.uuid4()),
            'tier': tier.value,
            'features': tier_config.features,
            'api_key': self._generate_api_key(tier),
            'monthly_cost': tier_config.price,
            'start_date': datetime.now(),
            'api_limits': {
                'daily_calls': tier_config.api_calls // 30,
                'monthly_calls': tier_config.api_calls
            }
        } 