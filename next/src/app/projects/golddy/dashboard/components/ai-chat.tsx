"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface Message {
  id: number;
  content: string;
  isBot: boolean;
  time: string;
  agentName?: string;
  agentColor?: string;
}

interface Agent {
  id: string;
  name: string;
  color: string;
  role: string;
  greeting: string;
  placeholder: string;
}

const agents: Agent[] = [
  {
    id: "meta",
    name: "Meta Agent",
    color: "#ef4444",
    role: "Coordinateur IA",
    greeting: "Je suis le Meta Agent, capable de coordonner les conversations entre les différents experts pour vous apporter une analyse complète.",
    placeholder: "Ex: Analyse complète de mon profil avec tous les experts...",
  },
  {
    id: "scraper",
    name: "Scraper",
    color: "#0ea5e9",
    role: "Expert en Scraping",
    greeting: "Je suis votre expert en scraping. Je peux analyser les données de vos concurrents et du marché pour vous donner des insights précieux.",
    placeholder: "Ex: Analyse les tendances de mes concurrents...",
  },
  // ... autres agents
];

export function AIChat() {
  const [currentAgent, setCurrentAgent] = useState<Agent>(agents[0]);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: currentAgent.greeting,
      isBot: true,
      time: new Date().toLocaleTimeString().slice(0, 5),
      agentName: currentAgent.role,
      agentColor: currentAgent.color,
    },
  ]);
  const [newMessage, setNewMessage] = useState("");

  const selectAgent = (agent: Agent) => {
    setCurrentAgent(agent);
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        content: agent.greeting,
        isBot: true,
        time: new Date().toLocaleTimeString().slice(0, 5),
        agentName: agent.role,
        agentColor: agent.color,
      },
    ]);
  };

  const sendMessage = async () => {
    if (!newMessage.trim()) return;

    const userMessage = newMessage;
    setNewMessage("");

    // Ajouter le message de l'utilisateur
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        content: userMessage,
        isBot: false,
        time: new Date().toLocaleTimeString().slice(0, 5),
      },
    ]);

    // Simuler une réponse du bot
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          content: `Réponse simulée pour : ${userMessage}`,
          isBot: true,
          time: new Date().toLocaleTimeString().slice(0, 5),
          agentName: currentAgent.role,
          agentColor: currentAgent.color,
        },
      ]);
    }, 1000);
  };

  return (
    <Card className="bg-gradient-to-br from-gray-900/50 to-gray-900/30 backdrop-blur-xl border-gray-800/50">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-lg text-white">Assistants IA</CardTitle>
        <div className="flex gap-2">
          {agents.map((agent) => (
            <Button
              key={agent.id}
              onClick={() => selectAgent(agent)}
              variant={currentAgent.id === agent.id ? "default" : "secondary"}
              className={cn(
                "px-3 py-1 rounded-full text-sm transition-colors",
                currentAgent.id === agent.id
                  ? "bg-blue-500 text-white hover:bg-blue-600"
                  : "bg-white/5 text-gray-400 hover:bg-white/10"
              )}
            >
              {agent.name}
            </Button>
          ))}
        </div>
      </CardHeader>

      <CardContent>
        <div className="h-[400px] flex flex-col">
          <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn("flex", message.isBot ? "justify-start" : "justify-end")}
              >
                <div
                  className={cn(
                    "max-w-[80%] rounded-lg p-3",
                    message.isBot ? "bg-gray-800/50" : "bg-blue-500/50"
                  )}
                >
                  {message.isBot && message.agentName && (
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className="text-xs font-medium"
                        style={{ color: message.agentColor }}
                      >
                        {message.agentName}
                      </span>
                    </div>
                  )}
                  <p className="text-sm text-gray-200">{message.content}</p>
                  <span className="text-xs text-gray-400 mt-1 block">
                    {message.time}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="flex gap-2">
            <Input
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder={currentAgent.placeholder}
              className="flex-1 bg-white/5 border border-gray-800 rounded-lg px-4 py-2 text-gray-300 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            />
            <Button
              onClick={sendMessage}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 