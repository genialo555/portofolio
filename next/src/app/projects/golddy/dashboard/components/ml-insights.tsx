"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface MLInsights {
  visibility: {
    increase: number;
    confidence: number;
  };
  content: {
    tips: string[];
  };
  growth: {
    engagement: number;
    confidence: number;
  };
}

interface MLInsightsProps {
  insights: MLInsights;
}

const mockInsights: MLInsights = {
  visibility: {
    increase: 15,
    confidence: 85
  },
  content: {
    tips: [
      "Utilisez plus de vidéos courtes",
      "Publiez aux heures de pointe",
      "Engagez avec des comptes similaires"
    ]
  },
  growth: {
    engagement: 23,
    confidence: 92
  }
};

export function MLInsights({ insights = mockInsights }: Partial<MLInsightsProps>) {
  return (
    <Card className="bg-gradient-to-br from-gray-900/50 to-gray-900/30 backdrop-blur-xl border-gray-800/50">
      <CardHeader>
        <CardTitle className="text-lg text-white">Insights IA</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-400">Optimisation de la visibilité</h3>
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-green-500 to-green-400 flex items-center justify-center">
              <span className="text-white text-sm font-medium">+{insights.visibility.increase}%</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-300">Confiance:</span>
              <span className="text-sm text-green-400">{insights.visibility.confidence}%</span>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-400">Conseils de contenu</h3>
          <ul className="space-y-2">
            {insights.content.tips.map((tip, index) => (
              <li key={index} className="flex items-center space-x-2">
                <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                <p className="text-sm text-gray-300">{tip}</p>
              </li>
            ))}
          </ul>
        </div>

        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-400">Stratégie de croissance</h3>
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-purple-400 flex items-center justify-center">
              <span className="text-white text-sm font-medium">+{insights.growth.engagement}%</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-300">Confiance:</span>
              <span className="text-sm text-purple-400">{insights.growth.confidence}%</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 