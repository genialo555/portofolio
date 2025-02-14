"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Pie } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  ChartData,
  ChartOptions
} from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

const engagementTypes = [
  { label: 'Likes', value: 65, color: '#3B82F6' },
  { label: 'Commentaires', value: 20, color: '#10B981' },
  { label: 'Partages', value: 10, color: '#8B5CF6' },
  { label: 'Sauvegardes', value: 5, color: '#F59E0B' }
];

const chartData: ChartData<"pie"> = {
  labels: engagementTypes.map(type => type.label),
  datasets: [{
    data: engagementTypes.map(type => type.value),
    backgroundColor: engagementTypes.map(type => type.color),
    borderWidth: 0
  }]
};

const chartOptions: ChartOptions<"pie"> = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false
    },
    tooltip: {
      backgroundColor: 'rgba(17, 24, 39, 0.8)',
      titleColor: '#fff',
      bodyColor: '#fff',
      padding: 12,
      borderColor: 'rgba(255, 255, 255, 0.1)',
      borderWidth: 1
    }
  }
};

export function EngagementPie() {
  return (
    <Card className="bg-gradient-to-br from-gray-900/50 to-gray-900/30 backdrop-blur-xl border-gray-800/50">
      <CardHeader>
        <CardTitle className="text-lg text-white">Engagement par type</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[300px] relative">
          <Pie data={chartData} options={chartOptions} />
        </div>
        <div className="mt-4 grid grid-cols-2 gap-4">
          {engagementTypes.map((item, index) => (
            <div key={index} className="flex items-center gap-2">
              <div 
                className="w-3 h-3 rounded-full" 
                style={{ backgroundColor: item.color }}
              />
              <span className="text-sm text-gray-400">
                {item.label}: {item.value}%
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
} 