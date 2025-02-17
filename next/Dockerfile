FROM node:18-alpine AS base

WORKDIR /app

# Install dependencies only when needed
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

# Copy all files
COPY . .

# Build the app
ENV NEXT_TELEMETRY_DISABLED 1

# Set API Keys
ENV NEXT_PUBLIC_GEMINI_API_KEY_POUR_1=AIzaSyAfpC2_s7EBfc6nskbG5Spx5ID4ZO4Rd88
ENV NEXT_PUBLIC_GEMINI_API_KEY_POUR_2=AIzaSyDvA5VuWhgiAf2JJsX13sspiJLPfeQf5QM
ENV NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1=AIzaSyATCNdzvw2LMS3huBrSDnPgrlOC5ASsa5s
ENV NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2=AIzaSyDKhGZqsCpxIrxNo2gE9T78pbOCdHjbUDw
ENV NEXT_PUBLIC_QWEN_API_KEY_1=517-1464c1f109dbb88f-1
ENV NEXT_PUBLIC_QWEN_API_KEY_2=517-1464c1f109dbb88f-2
ENV NEXT_PUBLIC_DEEPSEEK_API_KEY_1=517-1464c1f109dbb88f-3
ENV NEXT_PUBLIC_DEEPSEEK_API_KEY_2=517-1464c1f109dbb88f-4

RUN yarn build

# Remove development dependencies
RUN rm -rf node_modules
RUN yarn install --production --frozen-lockfile

# Set production environment variables
ENV NODE_ENV production
ENV PORT 8080
ENV HOSTNAME "0.0.0.0"

EXPOSE 8080

# Start the app
CMD ["yarn", "start"]
