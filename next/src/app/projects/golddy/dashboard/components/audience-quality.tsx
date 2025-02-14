"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface Metric {
  name: string;
  score: number;
}

interface AudienceQualityProps {
  data: {
    score: number;
    metrics: Metric[];
    tips: string[];
  };
}

export function AudienceQuality({ data }: AudienceQualityProps) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-400";
    if (score >= 60) return "text-yellow-400";
    return "text-red-400";
  };

  const getBarColor = (score: number) => {
    if (score >= 80) return "bg-gradient-to-r from-green-500 to-green-400";
    if (score >= 60) return "bg-gradient-to-r from-yellow-500 to-yellow-400";
    return "bg-gradient-to-r from-red-500 to-red-400";
  };

  return (
    <Card className="bg-gradient-to-br from-gray-900/50 to-gray-900/30 backdrop-blur-xl border-gray-800/50">
      <CardHeader>
        <CardTitle className="text-lg text-white">Qualité de l'audience</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Score global */}
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-32 h-32 rounded-full bg-gradient-to-r from-blue-500 to-purple-500">
            <div className="text-3xl font-bold text-white">{data.score}%</div>
          </div>
          <p className="mt-2 text-sm text-gray-400">Score de qualité global</p>
        </div>

        {/* Métriques détaillées */}
        <div className="grid gap-4">
          {data.metrics.map((metric) => (
            <div key={metric.name} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">{metric.name}</span>
                <span className={cn("text-sm font-medium", getScoreColor(metric.score))}>
                  {metric.score}%
                </span>
              </div>
              <Progress 
                value={metric.score} 
                className="h-2 bg-gray-700"
                indicatorClassName={getBarColor(metric.score)}
              />
            </div>
          ))}
        </div>

        {/* Recommandations */}
        <div className="p-4 bg-white/5 rounded-lg">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Recommandations</h3>
          <ul className="space-y-2">
            {data.tips.map((tip, index) => (
              <li key={index} className="flex items-start gap-2">
                <span className="text-green-400 mt-0.5">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth="2" 
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" 
                    />
                  </svg>
                </span>
                <span className="text-sm text-gray-400">{tip}</span>
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
} 