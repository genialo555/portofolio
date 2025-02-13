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
} from "@/components/ui/chat-bubble"
import { runDebate } from "./services/gemini-agents"
import { MediaToolbar } from "@/components/ui/media-toolbar"
import Image from "next/image"
import { ShimmerButton } from "@/components/ui/shimmer-button"
import { Vortex } from "@/components/ui/loading-vortex"

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

export default function ChatPage() {
  const [hovered, setHovered] = useState(false)
  const [topic, setTopic] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [currentDebate, setCurrentDebate] = useState<Debate | null>(null)
  const [selectedMedia, setSelectedMedia] = useState<MediaContent | null>(null)
  const [isGeneratingImage, setIsGeneratingImage] = useState(false)
  const [showVortex, setShowVortex] = useState(false)

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

  const handleNewDebate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!topic.trim() || isLoading) return
    
    setShowVortex(true)
    setIsLoading(true)

    try {
      // Attendre 5 secondes avec l'effet de vortex
      await new Promise(resolve => setTimeout(resolve, 5000))

      const newDebate: Debate = {
        id: crypto.randomUUID(),
        topic: topic.trim(),
        messages: [],
        lastUpdated: new Date()
      }
      setCurrentDebate(newDebate)

      // Lancer le débat
      const result = await runDebate(topic)

      // Ajouter les messages avec le média sélectionné
      const pourMessages: Message[] = result.pour.map(content => ({
        id: crypto.randomUUID(),
        content: content.replace(/\*/g, '').trim(),
        role: "pour",
        timestamp: new Date()
      }))

      const contreMessages: Message[] = result.contre.map(content => ({
        id: crypto.randomUUID(),
        content: content.replace(/\*/g, '').trim(),
        role: "contre",
        timestamp: new Date()
      }))

      const syntheseMessage: Message = {
        id: crypto.randomUUID(),
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

      <div
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        className="relative mx-auto w-full items-center justify-center overflow-hidden"
      >
        <div className="relative flex w-full items-center justify-center p-4">
          <AnimatePresence>
            {hovered && (
              <motion.div
                initial={{ opacity: 1 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 1 }}
                className="absolute inset-0 h-full w-full object-cover"
              >
                <CanvasRevealEffect
                  animationSpeed={5}
                  containerClassName="bg-transparent opacity-30 dark:opacity-50"
                  colors={[
                    [245, 5, 55],
                    [245, 5, 55],
                  ]}
                  opacities={[1, 0.8, 1, 0.8, 0.5, 0.8, 1, 0.5, 1, 3]}
                  dotSize={2}
                />
              </motion.div>
            )}
          </AnimatePresence>

          <div className="z-20 w-full max-w-7xl">
            <div className="px-6 py-8 text-center space-y-4">
              <h1 className="text-4xl font-bold tracking-tight">
                <TextGenerateEffect
                  words="Débat Multi-Agents avec Gemini"
                  className="bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent"
                />
              </h1>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                Entrez un sujet et laissez nos agents débattre et synthétiser les arguments.
              </p>
            </div>

            <div className="relative mt-4 px-4 mb-8">
              <form
                onSubmit={handleNewDebate}
                className="flex flex-col gap-2 max-w-2xl mx-auto"
              >
                <div className="flex gap-2">
                  <Input
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    placeholder="Entrez un sujet de débat..."
                    className="flex-1 h-12 px-4 text-lg"
                    disabled={isLoading}
                  />
                  <ShimmerButton
                    type="submit"
                    disabled={!topic.trim() || isLoading}
                    isLoading={isLoading}
                    className="min-w-[140px] font-medium"
                    shimmerColor="rgba(255, 255, 255, 0.1)"
                  >
                    {isLoading ? "Débat en cours..." : "Lancer le débat"}
                  </ShimmerButton>
                </div>

                <MediaToolbar
                  onImageSelect={handleImageSelect}
                  onVideoSelect={handleVideoSelect}
                  onImageGenerate={handleImageGenerate}
                  isGenerating={isGeneratingImage}
                />

                {selectedMedia && (
                  <div className="mt-2 relative">
                    <div className="relative w-full aspect-video rounded-lg overflow-hidden border">
                      {selectedMedia.type === "image" ? (
                        <Image
                          src={selectedMedia.url}
                          alt={selectedMedia.alt || "Image sélectionnée"}
                          fill
                          className="object-cover"
                        />
                      ) : (
                        <video
                          src={selectedMedia.url}
                          controls
                          className="w-full h-full"
                        />
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="absolute top-2 right-2 bg-background/80 hover:bg-background/90"
                        onClick={() => setSelectedMedia(null)}
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
                    </div>
                  </div>
                )}
              </form>
            </div>

            <div className="relative min-h-[calc(100vh-300px)]">
              {/* Colonnes de débat */}
              <div className="grid grid-cols-2 gap-8 px-4">
                {/* Colonne POUR */}
                <div className="space-y-4">
                  {pourMessages.length > 0 && (
                    <motion.h2
                      initial={{ opacity: 0, y: -20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="text-xl font-semibold text-green-700 dark:text-green-300 text-center mb-6"
                    >
                      Arguments POUR
                    </motion.h2>
                  )}
                  {pourMessages.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5 }}
                    >
                      <ChatBubble role="pour">
                        <ChatBubbleAvatar role="pour" fallback="P" />
                        <ChatBubbleMessage role="pour">
                          <div className="prose prose-sm max-w-none">
                            {message.content.split('\n').map((paragraph, i) => (
                              <p key={i} className="mb-2 last:mb-0">{paragraph}</p>
                            ))}
                          </div>
                        </ChatBubbleMessage>
                      </ChatBubble>
                    </motion.div>
                  ))}
                </div>

                {/* Colonne CONTRE */}
                <div className="space-y-4">
                  {contreMessages.length > 0 && (
                    <motion.h2
                      initial={{ opacity: 0, y: -20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="text-xl font-semibold text-red-700 dark:text-red-300 text-center mb-6"
                    >
                      Arguments CONTRE
                    </motion.h2>
                  )}
                  {contreMessages.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5 }}
                    >
                      <ChatBubble role="contre">
                        <ChatBubbleAvatar role="contre" fallback="C" />
                        <ChatBubbleMessage role="contre">
                          <div className="prose prose-sm max-w-none">
                            {message.content.split('\n').map((paragraph, i) => (
                              <p key={i} className="mb-2 last:mb-0">{paragraph}</p>
                            ))}
                          </div>
                        </ChatBubbleMessage>
                      </ChatBubble>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Synthèse au centre en bas */}
              {syntheseMessage && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                  className="relative mt-12 px-4"
                >
                  {/* Lignes de connexion */}
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
                      <ChatBubbleMessage role="synthese">
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

              {isLoading && (
                <div className="flex justify-center mt-8">
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
  )
} 