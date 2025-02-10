from typing import Dict, Any

class InstagramMLSalesFunnel:
    """Système de vente et onboarding."""
    
    async def generate_demo(self, profile_url: str) -> Dict[str, Any]:
        """Génère une démo gratuite personnalisée."""
        profile = await self.instagram_agent.analyze_profile(profile_url)
        
        return {
            'profile_analysis': {
                'current_metrics': profile.metrics,
                'growth_potential': profile.growth_forecast,
                'monetization_opportunities': profile.revenue_estimate
            },
            'sample_strategies': {
                'content': await self.instagram_agent.generate_sample_content(),
                'engagement': profile.engagement_recommendations,
                'growth': profile.growth_tactics[:3]  # Aperçu limité
            },
            'roi_projection': {
                'followers_3_months': profile.project_growth(months=3),
                'estimated_revenue': profile.estimate_revenue(),
                'roi_multiplier': profile.calculate_roi()
            }
        }
        
    async def create_proposal(self, profile_data: Dict) -> Dict[str, Any]:
        """Crée une proposition commerciale personnalisée."""
        metrics = await self.instagram_agent.analyze_profile(profile_data)
        
        recommended_tier = self._determine_best_tier(metrics)
        
        return {
            'recommended_plan': {
                'tier': recommended_tier,
                'price': self.pricing[recommended_tier].price,
                'roi_estimate': metrics.projected_roi,
                'custom_features': self._get_custom_features(metrics)
            },
            'value_proposition': {
                'current_challenges': metrics.identify_challenges(),
                'our_solutions': self._map_solutions_to_challenges(metrics),
                'expected_outcomes': metrics.project_outcomes(months=6)
            },
            'case_studies': await self._get_relevant_case_studies(metrics.niche),
            'next_steps': {
                'onboarding_process': self._get_onboarding_steps(recommended_tier),
                'timeline': self._generate_implementation_timeline(),
                'support_details': self.pricing[recommended_tier].support_level
            }
        } 