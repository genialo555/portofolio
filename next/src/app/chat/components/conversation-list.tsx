"use client"

import { Conversation } from "../types"
import { cn } from "@/lib/utils"
import { formatDistanceToNow } from "date-fns"
import { fr } from "date-fns/locale/fr"
import { motion } from "framer-motion"
import { ScrollArea } from "@/components/ui/scroll-area"

interface ConversationListProps {
  conversations: Conversation[]
  currentConversation: Conversation | null
  onSelectConversation: (conversation: Conversation) => void
  className?: string
}

export function ConversationList({
  conversations,
  currentConversation,
  onSelectConversation,
  className
}: ConversationListProps) {
  if (conversations.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-8">
        <div className="mb-2">👋</div>
        Commencez une nouvelle conversation
      </div>
    )
  }

  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex items-center justify-between px-2">
        <label className="text-sm font-medium text-foreground/60">
          Conversations
        </label>
        <span className="text-xs text-muted-foreground">
          {conversations.length} {conversations.length > 1 ? "débats" : "débat"}
        </span>
      </div>
      
      <ScrollArea className="h-[calc(100vh-200px)]">
        <div className="space-y-2 pr-4">
          {conversations.map((conversation, index) => (
            <motion.button
              key={conversation.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => onSelectConversation(conversation)}
              className={cn(
                "w-full p-3 text-left rounded-lg transition-all duration-200",
                "hover:bg-accent hover:text-accent-foreground",
                "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
                "border border-transparent",
                currentConversation?.id === conversation.id
                  ? "bg-accent text-accent-foreground border-primary/50"
                  : "text-foreground/80 hover:border-accent"
              )}
            >
              <div className="flex justify-between items-start gap-2">
                <div className="space-y-1 flex-1 min-w-0">
                  <p className="font-medium truncate">
                    {conversation.title}
                  </p>
                  <p className="text-xs text-muted-foreground truncate">
                    {conversation.messages[conversation.messages.length - 1]?.content || "Nouvelle conversation"}
                  </p>
                </div>
                <div className="flex flex-col items-end gap-1 shrink-0">
                  <span className="text-xs text-muted-foreground whitespace-nowrap">
                    {formatDistanceToNow(conversation.lastUpdated, {
                      addSuffix: true,
                      locale: fr
                    })}
                  </span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary">
                    {conversation.model}
                  </span>
                </div>
              </div>
            </motion.button>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
} 