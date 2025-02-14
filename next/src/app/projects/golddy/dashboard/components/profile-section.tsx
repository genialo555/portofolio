"use client";

import { Card, CardContent } from "@/components/ui/card";
import Image from "next/image";

interface User {
  fullName: string;
  username: string;
  avatar: string;
  bio: string;
  verified: boolean;
  followers: number;
  following: number;
  posts: number;
}

interface ProfileSectionProps {
  user: User;
}

const mockUser: User = {
  fullName: "Sarah Martin",
  username: "sarahmartin",
  avatar: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400",
  bio: "Créatrice de contenu lifestyle & bien-être | Conseils en développement personnel",
  verified: true,
  followers: 125000,
  following: 850,
  posts: 432
};

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K';
  }
  return num.toString();
}

export function ProfileSection({ user = mockUser }: Partial<ProfileSectionProps>) {
  return (
    <Card className="bg-gradient-to-br from-gray-900/50 to-gray-900/30 backdrop-blur-xl border-gray-800/50">
      <CardContent className="p-4">
        <div className="flex items-center space-x-4">
          <div className="relative w-16 h-16">
            <Image
              src={user.avatar}
              alt={user.fullName}
              fill
              className="rounded-full object-cover"
            />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-lg font-semibold text-white">{user.fullName}</h2>
              {user.verified && (
                <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              )}
            </div>
            <p className="text-sm text-gray-400">@{user.username}</p>
          </div>
        </div>
        <p className="mt-4 text-sm text-gray-300">{user.bio}</p>
        <div className="grid grid-cols-3 gap-4 mt-4">
          <div className="text-center">
            <div className="text-lg font-semibold text-white">{formatNumber(user.followers)}</div>
            <div className="text-xs text-gray-400">Abonnés</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-white">{formatNumber(user.following)}</div>
            <div className="text-xs text-gray-400">Abonnements</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-white">{formatNumber(user.posts)}</div>
            <div className="text-xs text-gray-400">Posts</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 