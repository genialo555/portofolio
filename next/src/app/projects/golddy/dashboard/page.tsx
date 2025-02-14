"use client";

import { AudienceQuality } from "./components/audience-quality";
import { AudienceStats } from "./components/audience-stats";
import { AIChat } from "./components/ai-chat";
import { EngagementPie } from "./components/engagement-pie";
import { FinancialSection } from "./components/financial-section";
import { GrowthChart } from "./components/growth-chart";
import { MLInsights } from "./components/ml-insights";
import { ProfileSection } from "./components/profile-section";
import { RecentPosts } from "./components/recent-posts";
import { HashtagAnalytics } from "./components/hashtag-analytics";
import Link from "next/link";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import Image from "next/image";

// Mock data
const mockAudienceQuality = {
  score: 85,
  realFollowers: 12500,
  engagement: 4.2,
  growth: 2.8,
  metrics: [
    { name: "Engagement authentique", score: 92 },
    { name: "Activité des followers", score: 78 },
    { name: "Croissance organique", score: 88 },
    { name: "Qualité des interactions", score: 82 },
  ],
  tips: [
    "Encouragez plus d'interactions dans les commentaires",
    "Publiez plus régulièrement aux heures de pointe",
    "Engagez-vous avec des comptes similaires",
  ],
};

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-black/95">
      {/* Bouton de retour */}
      <Link 
        href="/projects/golddy"
        className="fixed top-6 left-6 lg:top-8 lg:left-8 z-50 p-2 lg:p-3 rounded-full bg-white/90 shadow-lg hover:bg-white hover:scale-110 transition-all duration-200 group"
      >
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          width="24" 
          height="24" 
          viewBox="0 0 24 24" 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round" 
          className="transform group-hover:-translate-x-1 transition-transform duration-200 w-5 h-5 lg:w-6 lg:h-6"
        >
          <path d="M19 12H5M12 19l-7-7 7-7"/>
        </svg>
      </Link>

      {/* Console Frame */}
      <div className="container py-8">
        <motion.div 
          className="relative border border-gray-800 rounded-lg overflow-hidden bg-black/80 backdrop-blur-sm"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Console Header */}
          <div className="bg-gray-900/80 border-b border-gray-800 p-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <span className="text-gray-400 text-sm font-mono ml-3">Golddy Analytics Dashboard</span>
            </div>
            <div className="flex items-center gap-2 text-gray-500 text-sm font-mono">
              <span>Connected</span>
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            </div>
          </div>

          {/* Console Content */}
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Preview Image */}
              <motion.div 
                className="lg:col-span-3"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <Card className="bg-white/5 backdrop-blur-sm border-gray-800">
                  <CardContent className="p-6">
                    <div className="relative aspect-video w-full overflow-hidden rounded-lg">
                      <Image
                        src="/Screenshot from 2025-02-14 14-21-02.png"
                        alt="Golddy Home Page"
                        fill
                        className="object-cover"
                        quality={100}
                      />
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Profile Section */}
              <motion.div 
                className="lg:col-span-1"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <ProfileSection />
              </motion.div>

              {/* Growth Chart */}
              <motion.div 
                className="lg:col-span-2"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <GrowthChart />
              </motion.div>

              {/* Engagement Pie */}
              <motion.div 
                className="lg:col-span-1"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <EngagementPie />
              </motion.div>

              {/* Financial Section */}
              <motion.div 
                className="lg:col-span-2"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <FinancialSection kpis={{
                  revenue: {
                    sponsoredPosts: 5000,
                    affiliateMarketing: 3000,
                    productSales: 7500,
                    coaching: 4500
                  }
                }} />
              </motion.div>

              {/* ML Insights */}
              <motion.div 
                className="lg:col-span-1"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
              >
                <MLInsights />
              </motion.div>

              {/* Recent Posts */}
              <motion.div 
                className="lg:col-span-2"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
              >
                <RecentPosts />
              </motion.div>

              {/* Hashtag Analytics */}
              <motion.div 
                className="lg:col-span-3"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
              >
                <HashtagAnalytics />
              </motion.div>

              {/* Chat IA */}
              <motion.div 
                className="lg:col-span-3"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
              >
                <AIChat />
              </motion.div>
            </div>
          </div>

          {/* Console Footer */}
          <div className="bg-gray-900/80 border-t border-gray-800 p-3">
            <div className="flex items-center justify-between text-xs font-mono text-gray-500">
              <div className="flex items-center gap-4">
                <span>Status: Online</span>
                <span>Last update: {new Date().toLocaleTimeString()}</span>
              </div>
              <div className="flex items-center gap-2">
                <span>v1.0.0</span>
                <span>•</span>
                <span>Golddy Analytics</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
} 