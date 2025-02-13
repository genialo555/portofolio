"use client"

import { useState } from "react"
import { Conversation, Message } from "../page"
import { cn } from "@/lib/utils"
import { formatDistanceToNow } from "date-fns"
import { fr } from "date-fns/locale"
import { motion } from "framer-motion"

interface ChatWindowProps {
  conversation: Conversation
  onSendMessage: (content: string) => void
}

export function ChatWindow({ conversation, onSendMessage }: ChatWindowProps) {
  const [input, setInput] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    onSendMessage(input)
    setInput("")
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* En-tête */}
      <div className="border-b border-border p-4">
        <h2 className="font-semibold">{conversation.title}</h2>
        <p className="text-sm text-muted-foreground">
          Modèle : {conversation.model}
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {conversation.messages.map((message, index) => (
          <MessageBubble
            key={message.id}
            message={message}
            isLastMessage={index === conversation.messages.length - 1}
          />
        ))}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-border p-4">
        <div className="flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Tapez votre message..."
            className="flex-1 px-4 py-2 rounded-lg border border-input bg-background text-foreground"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            Envoyer
          </button>
        </div>
      </form>
    </div>
  )
}

function MessageBubble({ message, isLastMessage }: { message: Message; isLastMessage: boolean }) {
  const isUser = message.role === "user"

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "flex gap-4",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-[80%] rounded-lg p-4",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground",
          isLastMessage && "animate-in fade-in-50"
        )}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>
        <div
          className={cn(
            "mt-2 text-xs",
            isUser ? "text-primary-foreground/60" : "text-muted-foreground"
          )}
        >
          {formatDistanceToNow(message.timestamp, {
            addSuffix: true,
            locale: fr
          })}
        </div>
      </div>
    </motion.div>
  )
} 