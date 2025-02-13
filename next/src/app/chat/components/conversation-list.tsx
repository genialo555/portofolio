"use client"

import { Conversation } from "../page"
import { cn } from "@/lib/utils"
import { formatDistanceToNow } from "date-fns"
import { fr } from "date-fns/locale"

interface ConversationListProps {
  conversations: Conversation[]
  currentConversation: Conversation | null
  onSelectConversation: (conversation: Conversation) => void
}

export function ConversationList({
  conversations,
  currentConversation,
  onSelectConversation,
}: ConversationListProps) {
  if (conversations.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        Aucune conversation
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-foreground/60">
        Conversations
      </label>
      <div className="space-y-2">
        {conversations.map((conversation) => (
          <button
            key={conversation.id}
            onClick={() => onSelectConversation(conversation)}
            className={cn(
              "w-full p-3 text-left rounded-lg transition-colors",
              "hover:bg-accent hover:text-accent-foreground",
              "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
              currentConversation?.id === conversation.id
                ? "bg-accent text-accent-foreground"
                : "text-foreground/80"
            )}
          >
            <div className="flex justify-between items-start gap-2">
              <div className="space-y-1">
                <p className="font-medium line-clamp-1">
                  {conversation.title}
                </p>
                <p className="text-xs text-muted-foreground line-clamp-1">
                  {conversation.messages[conversation.messages.length - 1]?.content || "Nouvelle conversation"}
                </p>
              </div>
              <div className="flex flex-col items-end gap-1">
                <span className="text-xs text-muted-foreground">
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
          </button>
        ))}
      </div>
    </div>
  )
} 