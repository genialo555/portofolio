"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Image from "next/image";

interface Post {
  id: number;
  thumbnail: string;
  caption: string;
  likes: number;
  comments: number;
  date: string;
}

const recentPosts: Post[] = [
  {
    id: 1,
    thumbnail: 'https://images.unsplash.com/photo-1501854140801-50d01698950b?w=400',
    caption: 'Comment optimiser votre stratégie marketing sur Instagram en 2024 #marketing #socialmedia',
    likes: 245,
    comments: 18,
    date: 'Il y a 2h'
  },
  {
    id: 2,
    thumbnail: 'https://images.unsplash.com/photo-1520531158340-44015069e78e?w=400',
    caption: 'Les tendances à suivre pour augmenter votre engagement #trends #engagement',
    likes: 189,
    comments: 12,
    date: 'Il y a 5h'
  },
  {
    id: 3,
    thumbnail: 'https://images.unsplash.com/photo-1449495169669-7b118f960251?w=400',
    caption: 'Découvrez nos derniers conseils pour développer votre communauté #community #growth',
    likes: 321,
    comments: 24,
    date: 'Il y a 8h'
  }
];

export function RecentPosts() {
  return (
    <Card className="bg-gradient-to-br from-gray-900/50 to-gray-900/30 backdrop-blur-xl border-gray-800/50">
      <CardHeader>
        <CardTitle className="text-lg text-white">Posts Récents</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {recentPosts.map((post) => (
          <div
            key={post.id}
            className="flex items-start gap-4 p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
          >
            <div className="relative w-16 h-16 shrink-0">
              <Image
                src={post.thumbnail}
                alt={post.caption}
                fill
                className="object-cover rounded-lg"
              />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-300 truncate">{post.caption}</p>
              <div className="mt-2 flex items-center gap-4">
                <div className="flex items-center gap-1">
                  <span className="text-red-400">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" />
                    </svg>
                  </span>
                  <span className="text-sm text-gray-400">{post.likes}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-blue-400">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                      />
                    </svg>
                  </span>
                  <span className="text-sm text-gray-400">{post.comments}</span>
                </div>
                <span className="text-xs text-gray-500">{post.date}</span>
              </div>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
} 