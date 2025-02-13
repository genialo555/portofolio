"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { MessageLoading } from "@/components/ui/message-loading"
import { motion, AnimatePresence } from "framer-motion"
import Image from "next/image"

interface MediaContent {
  type: "image" | "video"
  url: string
  alt?: string
}

interface ChatBubbleProps {
  variant?: "sent" | "received"
  role?: "pour" | "contre" | "synthese"
  layout?: "default" | "ai"
  className?: string
  children: React.ReactNode
}

export function ChatBubble({
  variant = "received",
  role = "pour",
  layout = "default",
  className,
  children,
}: ChatBubbleProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn(
        "flex items-start gap-3 mb-4 group",
        variant === "sent" && "flex-row-reverse",
        className,
      )}
    >
      {children}
    </motion.div>
  )
}

interface ChatBubbleMessageProps {
  variant?: "sent" | "received"
  role?: "pour" | "contre" | "synthese"
  isLoading?: boolean
  media?: MediaContent
  className?: string
  children?: React.ReactNode
}

export function ChatBubbleMessage({
  variant = "received",
  role = "pour",
  isLoading,
  media,
  className,
  children,
}: ChatBubbleMessageProps) {
  const [isExpanded, setIsExpanded] = React.useState(false)
  const [isMediaLoaded, setIsMediaLoaded] = React.useState(false)
  const contentRef = React.useRef<HTMLDivElement>(null)
  const [hasOverflow, setHasOverflow] = React.useState(false)

  React.useEffect(() => {
    if (contentRef.current) {
      setHasOverflow(contentRef.current.scrollHeight > 100)
    }
  }, [children])

  return (
    <>
      <div
        className={cn(
          "relative transition-all duration-200 overflow-hidden font-sans",
          "max-w-[300px] rounded-full flex items-center justify-center p-6",
          "before:content-[''] before:absolute before:w-3 before:h-3 before:rotate-45",
          role === "pour" && [
            "bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-900/10 border border-green-200/50 dark:border-green-500/20",
            "before:bg-gradient-to-br before:from-green-50 before:to-green-100 dark:before:from-green-900/20 dark:before:to-green-900/10 before:border-l before:border-t before:border-green-200/50 dark:before:border-green-500/20",
            variant === "sent" ? "before:right-[-6px]" : "before:left-[-6px]"
          ],
          role === "contre" && [
            "bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-900/10 border border-red-200/50 dark:border-red-500/20",
            "before:bg-gradient-to-br before:from-red-50 before:to-red-100 dark:before:from-red-900/20 dark:before:to-red-900/10 before:border-l before:border-t before:border-red-200/50 dark:before:border-red-500/20",
            variant === "sent" ? "before:right-[-6px]" : "before:left-[-6px]"
          ],
          role === "synthese" && [
            "bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/10 border border-blue-200/50 dark:border-blue-500/20",
            "before:bg-gradient-to-br before:from-blue-50 before:to-blue-100 dark:before:from-blue-900/20 dark:before:to-blue-900/10 before:border-l before:border-t before:border-blue-200/50 dark:before:border-blue-500/20",
            "before:left-1/2 before:transform before:-translate-x-1/2 before:-bottom-2 before:rotate-[225deg]"
          ],
          variant === "sent" && "ml-auto",
          "group-hover:shadow-lg",
          className
        )}
      >
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <MessageLoading />
          </div>
        ) : (
          <div className="relative w-full">
            {media && (
              <div className="relative w-full aspect-video mb-2 rounded-full overflow-hidden">
                {media.type === "image" ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: isMediaLoaded ? 1 : 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Image
                      src={media.url}
                      alt={media.alt || "Image générée"}
                      fill
                      className="object-cover"
                      onLoad={() => setIsMediaLoaded(true)}
                    />
                    {!isMediaLoaded && (
                      <div className="absolute inset-0 flex items-center justify-center bg-muted/10 backdrop-blur-sm">
                        <MessageLoading />
                      </div>
                    )}
                  </motion.div>
                ) : (
                  <video
                    src={media.url}
                    controls
                    className="absolute inset-0 w-full h-full"
                    onLoadedData={() => setIsMediaLoaded(true)}
                  />
                )}
              </div>
            )}
            <div
              ref={contentRef}
              className={cn(
                "transition-all duration-200",
                !isExpanded && "max-h-[100px]",
                !isExpanded && hasOverflow && "overflow-hidden mask-bottom"
              )}
            >
              <div className="space-y-2 text-sm leading-relaxed tracking-tight">
                {typeof children === 'string' ? 
                  children.split('\n').map((paragraph, i) => (
                    <p key={i} className="text-sm text-foreground/90">
                      {paragraph}
                    </p>
                  )) : 
                  children
                }
              </div>
            </div>
            {hasOverflow && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsExpanded(true)}
                className={cn(
                  "absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 text-xs font-medium px-2 py-1 h-auto min-h-0 bg-white shadow-sm hover:shadow-md",
                  role === "pour" && "text-green-700 hover:text-green-800",
                  role === "contre" && "text-red-700 hover:text-red-800",
                  role === "synthese" && "text-blue-700 hover:text-blue-800"
                )}
              >
                Voir plus
              </Button>
            )}
          </div>
        )}
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
            onClick={() => setIsExpanded(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className={cn(
                "relative w-full max-w-lg p-6 rounded-xl shadow-lg",
                role === "pour" && "bg-green-50 dark:bg-green-900/20",
                role === "contre" && "bg-red-50 dark:bg-red-900/20",
                role === "synthese" && "bg-blue-50 dark:bg-blue-900/20"
              )}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="prose prose-sm max-w-none">
                {typeof children === 'string' ? 
                  children.split('\n').map((paragraph, i) => (
                    <p key={i} className="mb-4 last:mb-0">
                      {paragraph}
                    </p>
                  )) : 
                  children
                }
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsExpanded(false)}
                className="absolute top-2 right-2"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M18 6 6 18"/>
                  <path d="m6 6 12 12"/>
                </svg>
              </Button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

interface ChatBubbleAvatarProps {
  src?: string
  fallback?: string
  role?: "pour" | "contre" | "synthese"
  className?: string
}

export function ChatBubbleAvatar({
  src,
  fallback = "AI",
  role = "pour",
  className,
}: ChatBubbleAvatarProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, delay: 0.1 }}
    >
      <Avatar className={cn(
        "h-12 w-12 ring-2 ring-offset-2 transition-all duration-200",
        role === "pour" && "ring-green-500/50 bg-green-100 dark:bg-green-900/20",
        role === "contre" && "ring-red-500/50 bg-red-100 dark:bg-red-900/20",
        role === "synthese" && "ring-blue-500/50 bg-blue-100 dark:bg-blue-900/20",
        "group-hover:ring-offset-4",
        className
      )}>
        {src && <AvatarImage src={src} />}
        <AvatarFallback className={cn(
          "font-medium text-sm",
          role === "pour" && "text-green-700 dark:text-green-300",
          role === "contre" && "text-red-700 dark:text-red-300",
          role === "synthese" && "text-blue-700 dark:text-blue-300"
        )}>
          {fallback}
        </AvatarFallback>
      </Avatar>
    </motion.div>
  )
}

interface ChatBubbleActionProps {
  icon?: React.ReactNode
  onClick?: () => void
  className?: string
}

export function ChatBubbleAction({
  icon,
  onClick,
  className,
}: ChatBubbleActionProps) {
  return (
    <Button
      variant="ghost"
      size="icon"
      className={cn("h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity", className)}
      onClick={onClick}
    >
      {icon}
    </Button>
  )
}

export function ChatBubbleActionWrapper({
  className,
  children,
}: {
  className?: string
  children: React.ReactNode
}) {
  return (
    <div className={cn("flex items-center gap-2 mt-2 px-1 text-muted-foreground/60", className)}>
      {children}
    </div>
  )
} 