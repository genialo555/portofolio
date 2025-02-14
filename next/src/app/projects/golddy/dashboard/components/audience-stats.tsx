"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface AudienceStatsProps {
  data: {
    score: number;
    realFollowers: number;
    engagement: number;
    growth: number;
  };
}

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + "K";
  }
  return num.toString();
}

export function AudienceStats({ data }: AudienceStatsProps) {
  return (
    <Card className="bg-gradient-to-br from-gray-900/50 to-gray-900/30 backdrop-blur-xl border-gray-800/50">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-gray-400">
          Statistiques de l'audience
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Score global */}
        <div className="relative w-full h-4 bg-gray-800 rounded-full overflow-hidden mb-6">
          <div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-green-500 to-emerald-400"
            style={{ width: `${data.score}%` }}
          />
        </div>

        {/* Statistiques détaillées */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Vrais abonnés</span>
            <span className="text-sm font-medium text-white">
              {formatNumber(data.realFollowers)}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Taux d'engagement</span>
            <span className="text-sm font-medium text-white">
              {data.engagement}%
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Croissance mensuelle</span>
            <span className="text-sm font-medium text-green-400">
              +{data.growth}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 