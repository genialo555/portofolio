"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartData,
  ChartOptions
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface FollowersHistory {
  date: string;
  count: number;
}

interface GrowthPrediction {
  trend: 'up' | 'down' | 'stable';
  confidence: number;
}

interface GrowthChartProps {
  prediction: GrowthPrediction;
  history: FollowersHistory[];
}

// Mock data
const mockHistory: FollowersHistory[] = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString(),
  count: 1800 + Math.floor(Math.random() * 1000)
}));

const mockPrediction: GrowthPrediction = {
  trend: 'up',
  confidence: 85
};

export function GrowthChart({ 
  prediction = mockPrediction, 
  history = mockHistory 
}: Partial<GrowthChartProps>) {
  const chartData: ChartData<"line"> = {
    labels: history.map(h => new Date(h.date).toLocaleDateString('fr-FR', { 
      day: 'numeric',
      month: 'short'
    })),
    datasets: [
      {
        label: 'Croissance attendue',
        data: history.map(h => h.count),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2
      },
      {
        label: 'Moyenne du secteur',
        data: history.map(h => h.count * 0.8), // Simulé pour la démo
        borderColor: '#6b7280',
        backgroundColor: 'transparent',
        fill: false,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2,
        borderDash: [5, 5]
      }
    ]
  };

  const chartOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      intersect: false,
      mode: 'index'
    },
    scales: {
      y: {
        beginAtZero: false,
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
          drawBorder: false
        },
        ticks: {
          color: '#64748b',
          font: {
            size: 11
          },
          padding: 10,
          callback: (value) => {
            if (typeof value === 'number' && value >= 1000) {
              return (value / 1000).toFixed(0) + 'k';
            }
            return value;
          }
        }
      },
      x: {
        grid: {
          display: false,
          drawBorder: false
        },
        ticks: {
          color: '#64748b',
          font: {
            size: 11
          },
          padding: 10
        }
      }
    },
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#94a3b8',
        bodyColor: '#f8fafc',
        padding: 12,
        displayColors: false,
        callbacks: {
          title: (items) => items[0].label,
          label: (context) => {
            const label = context.dataset.label;
            const value = context.raw as number;
            return `${label}: ${value.toLocaleString()} abonnés`;
          }
        }
      }
    }
  };

  return (
    <Card className="bg-white/5 backdrop-blur-sm border-gray-800">
      <CardContent className="p-4">
        <div className="relative w-full h-[300px]">
          <Line data={chartData} options={chartOptions} />
        </div>
      </CardContent>
    </Card>
  );
} 