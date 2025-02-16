"use client"

import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { CanvasRevealEffect } from "@/components/ui/canvas-effect"
import { TextGenerateEffect } from "@/components/ui/text-generate-effect"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import {
  ChatBubble,
  ChatBubbleMessage,
  ChatBubbleAvatar,
  ChatBubbleAction,
  ChatBubbleActionWrapper,
  ChatBubbleMessageProps,
} from "./components/chat-bubble"
import { runDebate } from "./services/gemini-agents"
import { MediaToolbar } from "@/components/ui/media-toolbar"
import Image from "next/image"
import { ShimmerButton } from "@/components/ui/shimmer-button"
import { Vortex } from "@/components/ui/loading-vortex"
import Link from "next/link"
import { ModelSelector } from "./components/model-selector"
import { CrossModelConfig, DebateRole } from "./types"
import { DEFAULT_MODEL_CONFIG } from "./services/ai-models"

interface Message {
  id: string
  content: string
  role: "pour" | "contre" | "synthese"
  timestamp: Date
  media?: MediaContent
}

interface Debate {
  id: string
  topic: string
  messages: Message[]
  lastUpdated: Date
}

interface MediaContent {
  type: string
  url: string
  alt?: string
}

const generateUUID = () => {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
};

export default function ChatPage() {
  const [hovered, setHovered] = useState(false)
  const [topic, setTopic] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [currentDebate, setCurrentDebate] = useState<Debate | null>(null)
  const [selectedMedia, setSelectedMedia] = useState<MediaContent | null>(null)
  const [isGeneratingImage, setIsGeneratingImage] = useState(false)
  const [showVortex, setShowVortex] = useState(false)
  const [expandedMessages, setExpandedMessages] = useState<Set<string>>(new Set())
  const [selectedModels, setSelectedModels] = useState<CrossModelConfig>(DEFAULT_MODEL_CONFIG)

  const handleExpandMessage = (messageId: string) => {
    setExpandedMessages(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  };

  const handleImageSelect = async (file: File) => {
    const url = URL.createObjectURL(file)
    setSelectedMedia({
      type: "image",
      url,
      alt: file.name
    })
  }

  const handleVideoSelect = async (file: File) => {
    const url = URL.createObjectURL(file)
    setSelectedMedia({
      type: "video",
      url
    })
  }

  const handleImageGenerate = async () => {
    if (!topic.trim()) return
    setIsGeneratingImage(true)
    try {
      // Ici, vous pouvez appeler votre API de génération d'images
      // Pour l'exemple, on simule un délai
      await new Promise(resolve => setTimeout(resolve, 2000))
      // Simulons une URL d'image générée
      setSelectedMedia({
        type: "image",
        url: "https://source.unsplash.com/random",
        alt: "Image générée"
      })
    } catch (error) {
      console.error("Error generating image:", error)
    } finally {
      setIsGeneratingImage(false)
    }
  }

  const handleModelChange = (role: DebateRole, model: string) => {
    setSelectedModels(prev => ({
      ...prev,
      [role]: model
    }));
  };

  const handleNewDebate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!topic.trim() || isLoading) return
    
    setShowVortex(true)
    setIsLoading(true)

    try {
      // Attendre 5 secondes avec l'effet de vortex
      await new Promise(resolve => setTimeout(resolve, 5000))

      const newDebate: Debate = {
        id: generateUUID(),
        topic: topic.trim(),
        messages: [],
        lastUpdated: new Date()
      }
      setCurrentDebate(newDebate)

      // Lancer le débat
      const result = await runDebate(topic)

      // Ajouter les messages avec le média sélectionné
      const pourMessages: Message[] = result.pour.map(content => ({
        id: generateUUID(),
        content: content.replace(/\*/g, '').trim(),
        role: "pour",
        timestamp: new Date()
      }))

      const contreMessages: Message[] = result.contre.map(content => ({
        id: generateUUID(),
        content: content.replace(/\*/g, '').trim(),
        role: "contre",
        timestamp: new Date()
      }))

      const syntheseMessage: Message = {
        id: generateUUID(),
        content: result.synthese.replace(/\*/g, '').trim(),
        role: "synthese",
        timestamp: new Date()
      }

      // Ajouter le média au premier message si présent
      if (selectedMedia) {
        pourMessages[0].media = selectedMedia
      }

      const updatedDebate = {
        ...newDebate,
        messages: [...pourMessages, ...contreMessages, syntheseMessage],
        lastUpdated: new Date()
      }

      setCurrentDebate(updatedDebate)
      setTopic("")
      setSelectedMedia(null)
    } catch (error) {
      console.error("Error starting debate:", error)
    } finally {
      setShowVortex(false)
      setIsLoading(false)
    }
  }

  const pourMessages = currentDebate?.messages.filter(m => m.role === "pour") || []
  const contreMessages = currentDebate?.messages.filter(m => m.role === "contre") || []
  const syntheseMessage = currentDebate?.messages.find(m => m.role === "synthese")

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      {/* Bouton de retour */}
      <Link 
        href="/projects"
        className="fixed top-6 left-6 z-50 p-3 rounded-full bg-white/90 shadow-lg hover:bg-white hover:scale-110 transition-all duration-200 group"
      >
        <svg
          className="w-5 h-5 transform group-hover:-translate-x-1 transition-transform duration-200"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M10 19l-7-7m0 0l7-7m-7 7h18"
          />
        </svg>
      </Link>

      {/* Ajouter l'effet de vortex */}
      <AnimatePresence>
        {showVortex && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[9999] bg-background/80 backdrop-blur-sm flex items-center justify-center"
          >
            <div className="relative w-[300px] h-[300px]">
              <Vortex
                particleCount={500}
                baseHue={220}
                backgroundColor="transparent"
                baseSpeed={0.5}
                rangeSpeed={1}
                baseRadius={0.5}
                rangeRadius={1}
              >
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="text-lg font-medium text-foreground"
                  >
                    Préparation du débat...
                  </motion.div>
                </div>
              </Vortex>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="container py-8">
        <div className="max-w-5xl mx-auto">
          {/* Fenêtre de dialogue principale */}
          <div className="bg-card border rounded-xl shadow-lg overflow-hidden flex flex-col min-h-[80vh]">
            {/* En-tête */}
            <div className="border-b p-6">
              <div className="flex items-center justify-between">
                <h1 className="text-2xl font-semibold">Débat IA</h1>
                <div className="flex items-center gap-4">
                  <ModelSelector
                    selectedModels={selectedModels}
                    onModelChange={handleModelChange}
                  />
                </div>
              </div>
              
              {/* Formulaire de nouveau débat */}
              <form onSubmit={handleNewDebate} className="mt-6">
                <div className="flex gap-4">
                  <Input
                    type="text"
                    placeholder="Entrez un sujet de débat..."
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    className="flex-1"
                  />
                  <Button type="submit" disabled={!topic.trim() || isLoading}>
                    Démarrer
                  </Button>
                </div>
              </form>
            </div>

            {/* Zone de messages scrollable */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="space-y-8">
                {currentDebate && (
                  <div className="space-y-8">
                    {/* Messages Pour */}
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-4"
                    >
                      {currentDebate.messages
                        .filter((m) => m.role === "pour")
                        .map((message) => (
                          <ChatBubble key={message.id} role={message.role}>
                            <ChatBubbleAvatar role={message.role} fallback="P" />
                            <ChatBubbleMessage 
                              role={message.role}
                              expanded={expandedMessages.has(message.id)}
                              onExpand={() => handleExpandMessage(message.id)}
                            >
                              {message.content}
                            </ChatBubbleMessage>
                          </ChatBubble>
                        ))}
                    </motion.div>

                    {/* Messages Contre */}
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                      className="space-y-4"
                    >
                      {currentDebate.messages
                        .filter((m) => m.role === "contre")
                        .map((message) => (
                          <ChatBubble key={message.id} role={message.role}>
                            <ChatBubbleAvatar role={message.role} fallback="C" />
                            <ChatBubbleMessage 
                              role={message.role}
                              expanded={expandedMessages.has(message.id)}
                              onExpand={() => handleExpandMessage(message.id)}
                            >
                              {message.content}
                            </ChatBubbleMessage>
                          </ChatBubble>
                        ))}
                    </motion.div>

                    {/* Synthèse */}
                    {syntheseMessage && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4 }}
                        className="relative"
                      >
                        <div className="absolute inset-0 -top-8">
                          <svg className="w-full h-8" preserveAspectRatio="none">
                            <line
                              x1="25%"
                              y1="0"
                              x2="50%"
                              y2="100%"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeDasharray="4 4"
                              className="text-muted-foreground/30"
                            />
                            <line
                              x1="75%"
                              y1="0"
                              x2="50%"
                              y2="100%"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeDasharray="4 4"
                              className="text-muted-foreground/30"
                            />
                          </svg>
                        </div>

                        <div className="max-w-2xl mx-auto">
                          <ChatBubble role="synthese">
                            <ChatBubbleAvatar role="synthese" fallback="S" />
                            <ChatBubbleMessage 
                              role="synthese"
                              expanded={expandedMessages.has(syntheseMessage.id)}
                              onExpand={() => handleExpandMessage(syntheseMessage.id)}
                            >
                              <div className="font-semibold mb-2">Synthèse</div>
                              <div className="prose prose-sm max-w-none">
                                {syntheseMessage.content.split('\n').map((paragraph, i) => (
                                  <p key={i} className="mb-2 last:mb-0">{paragraph}</p>
                                ))}
                              </div>
                            </ChatBubbleMessage>
                          </ChatBubble>
                        </div>
                      </motion.div>
                    )}
                  </div>
                )}

                {isLoading && (
                  <div className="flex justify-center">
                    <ChatBubble>
                      <ChatBubbleAvatar fallback="..." />
                      <ChatBubbleMessage isLoading />
                    </ChatBubble>
                  </div>
                )}

                {!currentDebate && !isLoading && (
                  <div className="text-center py-12 text-muted-foreground">
                    Entrez un sujet pour démarrer un nouveau débat
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 