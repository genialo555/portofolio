"use client";

import { Card, CardContent } from "@/components/ui/card";

interface Hashtag {
  name: string;
  engagement: number;
  posts: number;
  category: string;
}

interface TopHashtag {
  name: string;
  engagement: number;
  posts: number;
}

interface TrendingHashtag {
  name: string;
  growth: number;
  description: string;
}

interface HashtagsData {
  top: TopHashtag[];
  trending: TrendingHashtag[];
}

// Mock data
const mockHashtags: Hashtag[] = [
  { name: 'marketing', engagement: 5.2, posts: 12500, category: 'Marketing Digital' },
  { name: 'digital', engagement: 4.8, posts: 8300, category: 'Marketing Digital' },
  { name: 'business', engagement: 4.5, posts: 7200, category: 'Business' },
  { name: 'socialmedia', engagement: 4.9, posts: 11000, category: 'Marketing Digital' },
  { name: 'entrepreneur', engagement: 3.8, posts: 5400, category: 'Business' },
  { name: 'success', engagement: 4.2, posts: 6800, category: 'Lifestyle' },
  { name: 'innovation', engagement: 5.1, posts: 9200, category: 'Business' },
  { name: 'strategy', engagement: 3.9, posts: 4800, category: 'Marketing Digital' },
  { name: 'growth', engagement: 4.7, posts: 7800, category: 'Business' },
  { name: 'leadership', engagement: 4.1, posts: 5600, category: 'Business' }
];

const mockTopHashtags: TopHashtag[] = [
  { name: 'marketing', engagement: 5.2, posts: 12.5 },
  { name: 'socialmedia', engagement: 4.9, posts: 11.0 },
  { name: 'innovation', engagement: 5.1, posts: 9.2 }
];

const mockTrendingHashtags: TrendingHashtag[] = [
  { name: 'digitalmarketing', growth: 45, description: 'Tendance croissante' },
  { name: 'contentcreator', growth: 32, description: 'Forte croissance' },
  { name: 'smm', growth: 28, description: 'En hausse' }
];

const getFontSize = (engagement: number): string => {
  const min = Math.min(...mockHashtags.map(h => h.engagement));
  const max = Math.max(...mockHashtags.map(h => h.engagement));
  const normalized = (engagement - min) / (max - min);
  const baseSize = 14; // taille minimum en pixels
  const maxSize = 28; // taille maximum en pixels
  return `${baseSize + normalized * (maxSize - baseSize)}px`;
};

const getColor = (category: string): string => {
  const colors: Record<string, string> = {
    'Marketing Digital': 'text-blue-400 hover:text-blue-300',
    'Business': 'text-purple-400 hover:text-purple-300',
    'Lifestyle': 'text-green-400 hover:text-green-300'
  };
  return colors[category] || 'text-gray-400 hover:text-gray-300';
};

export function HashtagAnalytics() {
  return (
    <Card className="bg-white/5 backdrop-blur-sm border-gray-800">
      <CardContent className="p-6">
        <h3 className="text-lg font-semibold text-white mb-6">Analyse des Hashtags</h3>
        
        <div className="space-y-6">
          {/* Hashtag Cloud */}
          <div>
            <h4 className="text-gray-400 mb-4">Nuage de Hashtags</h4>
            <div className="p-4 rounded-lg bg-white/5 min-h-[200px] flex flex-wrap gap-3 items-center justify-center">
              {mockHashtags.map((hashtag) => (
                <span
                  key={hashtag.name}
                  style={{ fontSize: getFontSize(hashtag.engagement) }}
                  className={`${getColor(hashtag.category)} transition-colors cursor-pointer`}
                >
                  #{hashtag.name}
                </span>
              ))}
            </div>
          </div>

          {/* Top Performing Hashtags */}
          <div>
            <h4 className="text-gray-400 mb-4">Top Hashtags</h4>
            <div className="space-y-4">
              {mockTopHashtags.map((hashtag) => (
                <div key={hashtag.name} className="flex items-center gap-3 p-2 rounded bg-gray-800/30">
                  <div className="shrink-0 w-8 h-8 rounded bg-gray-800/50 flex items-center justify-center">
                    <span className="text-sm text-gray-400">#</span>
                  </div>
                  <div className="flex-grow">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">{hashtag.name}</span>
                      <span className="text-xs text-gray-400">{hashtag.engagement}% eng.</span>
                    </div>
                    <div className="text-xs text-gray-400">{hashtag.posts}k posts</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Hashtag Categories */}
          <div>
            <h4 className="text-gray-400 mb-4">Catégories</h4>
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-white/5">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-white">Marketing Digital</span>
                  <span className="text-sm text-green-400">4.9% eng.</span>
                </div>
                <div className="flex flex-wrap gap-2 mt-3">
                  <span className="text-xs bg-white/10 text-gray-300 px-2 py-1 rounded-full">#marketing</span>
                  <span className="text-xs bg-white/10 text-gray-300 px-2 py-1 rounded-full">#digital</span>
                  <span className="text-xs bg-white/10 text-gray-300 px-2 py-1 rounded-full">#business</span>
                </div>
              </div>

              <div className="p-4 rounded-lg bg-white/5">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-white">Lifestyle</span>
                  <span className="text-sm text-green-400">4.2% eng.</span>
                </div>
                <div className="flex flex-wrap gap-2 mt-3">
                  <span className="text-xs bg-white/10 text-gray-300 px-2 py-1 rounded-full">#lifestyle</span>
                  <span className="text-xs bg-white/10 text-gray-300 px-2 py-1 rounded-full">#inspiration</span>
                  <span className="text-xs bg-white/10 text-gray-300 px-2 py-1 rounded-full">#motivation</span>
                </div>
              </div>
            </div>
          </div>

          {/* Trending Hashtags */}
          <div>
            <h4 className="text-gray-400 mb-4">Hashtags Tendance</h4>
            <div className="space-y-4">
              {mockTrendingHashtags.map((hashtag) => (
                <div key={hashtag.name} className="flex items-center gap-3 p-2 rounded bg-gray-800/30">
                  <div className="shrink-0 w-8 h-8 rounded bg-gray-800/50 flex items-center justify-center">
                    <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                  </div>
                  <div className="flex-grow">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">{hashtag.name}</span>
                      <span className="text-xs text-green-400">+{hashtag.growth}%</span>
                    </div>
                    <div className="text-xs text-gray-400">{hashtag.description}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <style jsx>{`
          .overflow-y-auto::-webkit-scrollbar {
            width: 4px;
          }

          .overflow-y-auto::-webkit-scrollbar-track {
            background: transparent;
          }

          .overflow-y-auto::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
          }

          .overflow-y-auto::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
          }
        `}</style>
      </CardContent>
    </Card>
  );
} 