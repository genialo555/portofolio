# Stage de build
FROM node:20-alpine AS builder

# Définition du répertoire de travail
WORKDIR /app/next

# Installation de Yarn
RUN apk add --no-cache yarn

# Copie des fichiers de configuration
COPY package.json yarn.lock ./
COPY tsconfig.json ./
COPY next.config.ts ./
COPY tailwind.config.ts ./
COPY postcss.config.mjs ./

# Installation des dépendances
RUN yarn install --frozen-lockfile

# Copie des fichiers source
COPY src ./src
COPY public ./public

# Vérification de la structure
RUN ls -la && \
    ls -la src && \
    ls -la src/app

# Variables d'environnement pour le build
ENV NEXT_TELEMETRY_DISABLED 1
ENV NEXT_PUBLIC_SITE_URL=https://jeremie-nunez.com
ENV NEXT_PUBLIC_API_URL=https://api.jeremie-nunez.com

# Variables pour Gemini (client)
ARG POUR_1
ARG POUR_2
ARG CONTRE_1
ARG CONTRE_2
ENV NEXT_PUBLIC_GEMINI_API_KEY_POUR_1=$POUR_1
ENV NEXT_PUBLIC_GEMINI_API_KEY_POUR_2=$POUR_2
ENV NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1=$CONTRE_1
ENV NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2=$CONTRE_2

# Variables pour Gemini (serveur)
ARG GEMINI_POUR_1
ARG GEMINI_POUR_2
ARG GEMINI_CONTRE_1
ARG GEMINI_CONTRE_2
ENV GEMINI_API_KEY_POUR=$GEMINI_POUR_1
ENV GEMINI_API_KEY_CONTRE=$GEMINI_CONTRE_1
ENV GEMINI_API_KEY=$GEMINI_POUR_1
ENV GEMINI_API_KEY_SYNTHESE=$GEMINI_POUR_1

# Variables pour Qwen (client)
ARG QWEN_1
ARG QWEN_2
ENV NEXT_PUBLIC_QWEN_API_KEY_1=$QWEN_1
ENV NEXT_PUBLIC_QWEN_API_KEY_2=$QWEN_2

# Variables pour Qwen (serveur)
ARG QWEN_API_1
ARG QWEN_API_2
ENV QWEN_API_KEY_1=$QWEN_API_1
ENV QWEN_API_KEY_2=$QWEN_API_2

# Variables pour DeepSeek (client)
ARG DEEPSEEK_1
ARG DEEPSEEK_2
ENV NEXT_PUBLIC_DEEPSEEK_API_KEY_1=$DEEPSEEK_1
ENV NEXT_PUBLIC_DEEPSEEK_API_KEY_2=$DEEPSEEK_2

# Variables pour DeepSeek (serveur)
ARG DEEPSEEK_API_1
ARG DEEPSEEK_API_2
ENV DEEPSEEK_API_KEY_1=$DEEPSEEK_API_1
ENV DEEPSEEK_API_KEY_2=$DEEPSEEK_API_2

# Build de l'application
RUN yarn build

# Stage de production
FROM node:20-alpine AS runner

WORKDIR /app/next

# Installation de Yarn
RUN apk add --no-cache yarn

# Variables d'environnement de production
ENV NODE_ENV=production
ENV PORT=8080
ENV NEXT_TELEMETRY_DISABLED=1
ENV HOSTNAME=0.0.0.0
ENV NEXT_PUBLIC_SITE_URL=https://jeremie-nunez.com
ENV NEXT_PUBLIC_API_URL=https://api.jeremie-nunez.com

# Variables pour Gemini (client)
ARG POUR_1
ARG POUR_2
ARG CONTRE_1
ARG CONTRE_2
ENV NEXT_PUBLIC_GEMINI_API_KEY_POUR_1=$POUR_1
ENV NEXT_PUBLIC_GEMINI_API_KEY_POUR_2=$POUR_2
ENV NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1=$CONTRE_1
ENV NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2=$CONTRE_2

# Variables pour Gemini (serveur)
ARG GEMINI_POUR_1
ARG GEMINI_POUR_2
ARG GEMINI_CONTRE_1
ARG GEMINI_CONTRE_2
ENV GEMINI_API_KEY_POUR=$GEMINI_POUR_1
ENV GEMINI_API_KEY_CONTRE=$GEMINI_CONTRE_1
ENV GEMINI_API_KEY=$GEMINI_POUR_1
ENV GEMINI_API_KEY_SYNTHESE=$GEMINI_POUR_1

# Variables pour Qwen (client)
ARG QWEN_1
ARG QWEN_2
ENV NEXT_PUBLIC_QWEN_API_KEY_1=$QWEN_1
ENV NEXT_PUBLIC_QWEN_API_KEY_2=$QWEN_2

# Variables pour Qwen (serveur)
ARG QWEN_API_1
ARG QWEN_API_2
ENV QWEN_API_KEY_1=$QWEN_API_1
ENV QWEN_API_KEY_2=$QWEN_API_2

# Variables pour DeepSeek (client)
ARG DEEPSEEK_1
ARG DEEPSEEK_2
ENV NEXT_PUBLIC_DEEPSEEK_API_KEY_1=$DEEPSEEK_1
ENV NEXT_PUBLIC_DEEPSEEK_API_KEY_2=$DEEPSEEK_2

# Variables pour DeepSeek (serveur)
ARG DEEPSEEK_API_1
ARG DEEPSEEK_API_2
ENV DEEPSEEK_API_KEY_1=$DEEPSEEK_API_1
ENV DEEPSEEK_API_KEY_2=$DEEPSEEK_API_2

# Copie des fichiers nécessaires depuis le builder
COPY --from=builder /app/next/next.config.ts ./
COPY --from=builder /app/next/tailwind.config.ts ./
COPY --from=builder /app/next/postcss.config.mjs ./
COPY --from=builder /app/next/public ./public
COPY --from=builder /app/next/package.json ./
COPY --from=builder /app/next/yarn.lock ./
COPY --from=builder /app/next/.next/standalone ./
COPY --from=builder /app/next/.next/static ./.next/static
COPY --from=builder /app/next/src ./src

# Installation des dépendances de production uniquement
RUN yarn install --production --frozen-lockfile

# Exposition du port
EXPOSE 8080

# Commande de démarrage
CMD ["yarn", "start"]
