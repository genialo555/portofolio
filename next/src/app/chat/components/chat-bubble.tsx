"use client";

import * as React from "react";
import Image from "next/image";
import { cn } from "@/lib/utils";
import { MessageLoading } from "@/components/ui/message-loading";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

interface MediaContent {
  type: "image" | "video";
  url: string;
  alt?: string;
}

export interface ChatBubbleMessageProps {
  variant?: "sent" | "received";
  role?: "pour" | "contre" | "synthese";
  isLoading?: boolean;
  media?: MediaContent;
  className?: string;
  children?: React.ReactNode;
  expanded?: boolean;
  onExpand?: () => void;
}

export interface ChatBubbleProps {
  variant?: "sent" | "received";
  role?: "pour" | "contre" | "synthese";
  layout?: "default" | "ai";
  className?: string;
  children: React.ReactNode;
}

export function ChatBubble({
  variant = "received",
  role = "pour",
  layout = "default",
  className,
  children,
}: ChatBubbleProps) {
  return (
    <div
      className={cn(
        "flex items-start gap-3",
        layout === "ai" && "flex-row-reverse",
        className
      )}
    >
      {children}
    </div>
  );
}

interface SynthesisContent {
  props: {
    className?: string;
    children: React.ReactNode[];
  };
}

export function ChatBubbleMessage({
  variant = "received",
  role = "pour",
  isLoading,
  media,
  className,
  children,
  expanded = false,
  onExpand,
}: ChatBubbleMessageProps) {
  const contentRef = React.useRef<HTMLDivElement>(null);
  const [hasOverflow, setHasOverflow] = React.useState(false);

  // Vérifier si le contenu dépasse la hauteur maximale
  React.useEffect(() => {
    if (contentRef.current) {
      const hasContentOverflow = contentRef.current.scrollHeight > 150;
      setHasOverflow(hasContentOverflow);
    }
  }, [children]);

  return (
    <div
      className={cn(
        "relative flex flex-col gap-2 w-full max-w-[85%] p-4 rounded-2xl",
        "transition-all duration-200",
        expanded ? "max-w-[95%]" : "max-w-[85%]",
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
      <div 
        ref={contentRef}
        className={cn(
          "overflow-y-auto pr-2 whitespace-pre-wrap break-words",
          expanded ? "max-h-[60vh]" : "max-h-[150px]"
        )}
      >
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

      {/* Expand/Collapse button */}
      {hasOverflow && (
        <button
          onClick={onExpand}
          className="mt-2 text-xs text-primary hover:text-primary/80 transition-colors"
        >
          {expanded ? "Voir moins" : "Voir plus"}
        </button>
      )}

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

export interface ChatBubbleAvatarProps {
  src?: string;
  fallback?: string;
  role?: "pour" | "contre" | "synthese";
  className?: string;
}

export function ChatBubbleAvatar({
  src,
  fallback = "AI",
  role = "pour",
  className,
}: ChatBubbleAvatarProps) {
  return (
    <Avatar
      className={cn(
        "h-8 w-8",
        role === "pour" && "bg-blue-500/20",
        role === "contre" && "bg-red-500/20",
        role === "synthese" && "bg-green-500/20",
        className
      )}
    >
      <AvatarFallback
        className={cn(
          role === "pour" && "text-blue-500",
          role === "contre" && "text-red-500",
          role === "synthese" && "text-green-500"
        )}
      >
        {fallback}
      </AvatarFallback>
    </Avatar>
  );
}

export interface ChatBubbleActionProps {
  icon?: React.ReactNode;
  onClick?: () => void;
  className?: string;
}

export function ChatBubbleAction({
  icon,
  onClick,
  className,
}: ChatBubbleActionProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "p-2 rounded-full hover:bg-muted/50 transition-colors",
        className
      )}
    >
      {icon}
    </button>
  );
}

export function ChatBubbleActionWrapper({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("flex items-center gap-1 mt-2", className)}>
      {children}
    </div>
  );
} 