"use client";

import Image from "next/image";
import { cn } from "@/lib/utils";
import { MessageLoading } from "@/components/ui/message-loading";

interface MediaContent {
  type: "image" | "video";
  url: string;
  alt?: string;
}

interface ChatBubbleMessageProps {
  variant?: "sent" | "received";
  role?: "pour" | "contre" | "synthese";
  isLoading?: boolean;
  media?: MediaContent;
  className?: string;
  children?: React.ReactNode;
}

export function ChatBubbleMessage({
  variant = "received",
  role = "pour",
  isLoading,
  media,
  className,
  children,
}: ChatBubbleMessageProps) {
  return (
    <div
      className={cn(
        "relative flex flex-col gap-2 w-full max-w-[85%] p-4 rounded-2xl",
        "transition-colors duration-200",
        variant === "sent"
          ? "ml-auto bg-primary text-primary-foreground"
          : cn(
              "bg-muted",
              role === "pour" && "bg-blue-500/10",
              role === "contre" && "bg-red-500/10",
              role === "synthese" && "bg-green-500/10"
            ),
        className
      )}
    >
      {/* Message content with max height and scrollbar */}
      <div className="max-h-[calc(100vh-200px)] overflow-y-auto pr-2 whitespace-pre-wrap break-words">
        <style jsx>{`
          div::-webkit-scrollbar {
            width: 6px;
          }
          div::-webkit-scrollbar-track {
            background: transparent;
          }
          div::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
          }
          div::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
          }
        `}</style>
        {isLoading ? (
          <MessageLoading />
        ) : (
          <div className="text-sm md:text-base">{children}</div>
        )}
      </div>

      {/* Media content if present */}
      {media && (
        <div className="mt-2">
          {media.type === "image" && (
            <div className="relative aspect-video w-full overflow-hidden rounded-lg">
              <Image
                src={media.url}
                alt={media.alt || "Image"}
                fill
                className="object-cover"
              />
            </div>
          )}
          {media.type === "video" && (
            <video
              src={media.url}
              controls
              className="w-full rounded-lg"
              style={{ maxHeight: "300px" }}
            />
          )}
        </div>
      )}
    </div>
  );
} 