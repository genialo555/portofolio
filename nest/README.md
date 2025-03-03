# RAG/KAG Hybrid System

<p align="center">
  <img src="https://via.placeholder.com/200x200?text=RAG+KAG" alt="RAG/KAG Logo" width="200" />
</p>

    <p align="center">
  <a href="#"><img src="https://img.shields.io/badge/nestjs-%3E=10.0.0-red.svg" alt="NestJS"></a>
  <a href="#"><img src="https://img.shields.io/badge/typescript-4.x-blue.svg" alt="TypeScript"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

## Overview

The RAG/KAG Hybrid System is a NestJS-based application that combines Retrieval Augmented Generation (RAG) and Knowledge Augmented Generation (KAG) approaches to deliver high-quality, contextually relevant responses to user queries. By orchestrating multiple specialized agent pools and implementing a debate protocol, the system produces responses that leverage both external knowledge sources and internal model capabilities.

## Key Features

- **Hybrid Intelligence**: Combines RAG's retrieval capabilities with KAG's internal knowledge processing
- **Multi-Agent Architecture**: Uses specialized agent pools (Commercial, Marketing, Sectorial) to analyze queries from different perspectives
- **Dialectical Debate System**: Confronts RAG and KAG analyses to produce optimal responses
- **Anomaly Detection**: System to identify and handle inconsistencies in responses
- **Expertise Level Adaptation**: Tailors responses based on the recipient's expertise level
- **API Flexibility**: Integration with multiple LLM providers (Google AI, Qwen, DeepSeek)
- **Resilient Processing**: Circuit breakers and error handling mechanisms
- **Educational Content**: Specialized agents for educational content with DeepSeek R1
- **Model Distillation**: Teaching-based system for training lightweight models

## System Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Application                       │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                          API Gateway                         │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                        Orchestrator                          │
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │ Router      │────▶│ Pool Manager│────▶│ Output      │    │
│  │             │     │             │     │ Collector   │    │
│  └─────────────┘     └─────────────┘     └─────────────┘    │
└───────────────────────────────┬─────────────────────────────┘
                                │
           ┌───────────────────┐│┌─────────────────┐
           │                   ││                  │
┌──────────▼───────┐  ┌────────▼▼─────────┐ ┌─────▼──────────┐
│                  │  │                   │ │                │
│  Agent Pools     │  │  Debate System    │ │  Synthesis     │
│  ┌────────────┐  │  │  ┌───────────┐   │ │                 │
│  │Commercial  │  │  │  │KAG Engine │   │ │                 │
│  └────────────┘  │  │  └───────────┘   │ │                 │
│  ┌────────────┐  │  │  ┌───────────┐   │ │                 │
│  │Marketing   │──┼──┼─▶│Debate     │───┼─┼▶                │
│  └────────────┘  │  │  │Protocol   │   │ │                 │
│  ┌────────────┐  │  │  └───────────┘   │ │                 │
│  │Sectorial   │  │  │  ┌───────────┐   │ │                 │
│  └────────────┘  │  │  │RAG Engine │   │ │                 │
│  ┌────────────┐  │  │  └───────────┘   │ │                 │
│  │Educational │  │  │                   │ │                 │
│  └────────────┘  │  │                   │ │                 │
└──────────────────┘  └───────────────────┘ └─────────────────┘
```

## Getting Started

### Prerequisites

- Node.js
- Yarn or npm
- API keys for LLM providers (if connecting to live services)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-organization/rag-kag-hybrid.git
cd rag-kag-hybrid
```

2. Install dependencies:
```bash
yarn install
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Build the application:
```bash
yarn build
```

### Running the Application

#### Development Mode
```bash
yarn start:dev
```

#### Production Mode
```bash
yarn start:prod
```

## API Documentation

Once the application is running, you can access the Swagger API documentation at:

```
http://localhost:3001/api/docs
```

### Main Endpoints

- `POST /api/rag-kag/query` - Process a user query
- `GET /api/rag-kag/health` - Check system health
- `POST /api/rag-kag/train/:modelName` - Force training for a specific model
- `GET /api/rag-kag/train/stats` - Get model training statistics

### API Usage Example

#### Process a Query with cURL

```bash
curl -X POST http://localhost:3001/api/rag-kag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main differences between RAG and KAG approaches?",
    "expertiseLevel": "ADVANCED",
    "useSimplifiedProcess": false
  }'
```

#### Force Model Training with cURL

```bash
curl -X POST http://localhost:3001/api/rag-kag/train/phi-3-mini \
  -H "Content-Type: application/json"
```

## Query Processing Flow

1. **Query Reception**: System receives and validates the user query
2. **Router Analysis**: Determines which agent pools are relevant for the query
3. **Parallel Processing**: Executes agents across the relevant pools
4. **Dual Analysis**: Processes the query through both RAG and KAG engines
5. **Debate Protocol**: Confronts the analyses to identify consensus and resolve contradictions
6. **Synthesis**: Generates a final, coherent response based on debate results
7. **Response Delivery**: Returns a structured response with metadata to the client

## House Model System

The system integrates a local model infrastructure that combines multiple open-source models for different tasks:

### Available Models

- **DeepSeek R1**: Specialized teacher model for educational content
- **Phi-3-mini**: Lightweight model for fast responses to simple queries
- **Llama-3-8B**: Balanced general-purpose model
- **Mistral-7B-FR**: French-specialized model for French language queries

### Model Distillation Process

The system implements a continuous learning architecture where:

1. **Example Collection**: DeepSeek R1 (teacher model) generates high-quality responses to queries
2. **Learning Repository**: The system stores these examples categorized by domain and query type
3. **Scheduled Training**: Distilled models (Phi-3-mini, Llama-3-8B, Mistral-7B-FR) are fine-tuned on these examples
4. **Expertise Transfer**: Over time, lightweight models learn to replicate the teacher model's expertise

### Automatic Learning

- **Runtime Learning**: The system automatically identifies cases where a distilled model lacks expertise
- **Asynchronous Teaching**: DeepSeek R1 processes these cases in the background
- **Continuous Improvement**: A scheduled task runs every 12 hours to update the models
- **Knowledge Specialization**: Models automatically specialize based on the query types they process

### Automated Evaluation System

The system includes a sophisticated evaluation framework for measuring and tracking distilled model performance:

- **Metrics**: Models are evaluated using BLEU, ROUGE, and semantic similarity metrics
- **Teacher Comparison**: Each model's outputs are compared against the DeepSeek R1 teacher model
- **Domain Expertise**: The system identifies which domains each model performs best in
- **Reliability Assessment**: Models are only used for production when they reach reliability thresholds
- **Smart Routing**: Queries are automatically routed to the most capable model based on content and language

### Monitoring and Management

- **Training Statistics**: Track the number of learning examples and model performance
- **Force Training**: Manually trigger training for specific models through the API
- **Evaluation Endpoints**: Access model evaluation metrics and reliability assessments
- **Domain Specialization**: View which domains each model excels in
- **Performance Analytics**: Monitor how models improve over time

### Intelligent Model Selection

The system employs a sophisticated model selection algorithm that:

1. **Analyzes Query Content**: Identifies the domain, language, and complexity of each query
2. **Evaluates Model Reliability**: Considers each model's proven reliability scores in relevant domains
3. **Domain Matching**: Matches query domains with models specialized in those areas
4. **Language Detection**: Routes French queries to French-specialized models when reliable
5. **Resource Optimization**: Uses lightweight models for simple queries when they meet reliability thresholds
6. **Guaranteed Quality**: Falls back to the teacher model when specialized models aren't sufficiently reliable

This intelligent routing system ensures optimal response quality while progressively utilizing more efficient models as they improve through continuous learning.

### New API Endpoints

Beyond the existing endpoints, the system now provides:

- `GET /api/rag-kag/evaluation/stats` - Get evaluation metrics for all models
- `POST /api/rag-kag/evaluation/:modelName` - Trigger evaluation for a specific model
- `GET /api/rag-kag/evaluation/:modelName/reliability` - Check if a model is reliable for production
- `GET /api/rag-kag/resilience/status` - View the status of all circuit breakers
- `POST /api/rag-kag/resilience/reset/:serviceName` - Reset a specific circuit breaker
- `POST /api/rag-kag/query/direct` - Direct query with anomaly detection & circuit breaker protection
- `POST /api/rag-kag/anomalies/detect` - Detect anomalies in any text content
- `GET /api/rag-kag/health/detailed` - Get detailed system health information

### API Usage Examples

#### Direct API Query with Anomaly Detection

```bash
curl -X POST http://localhost:3001/api/rag-kag/query/direct \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "GOOGLE_AI",
    "prompt": "What are the main differences between RAG and KAG approaches?",
    "detectAnomalies": true,
    "anomalyDetectionLevel": "MEDIUM_AND_ABOVE",
    "fallbackProvider": "HOUSE_MODEL"
  }'
```

#### Detecting Anomalies in Content

```bash
curl -X POST http://localhost:3001/api/rag-kag/anomalies/detect \
  -H "Content-Type: application/json" \
  -d '{
    "content": "All experts agree that AI is always dangerous. There is never a case where AI is beneficial.",
    "level": "ALL"
  }'
```

#### Get Detailed System Health

```bash
curl -X GET http://localhost:3001/api/rag-kag/health/detailed
```

## Advanced Architecture Patterns

### Circular Dependency Management

The system employs advanced NestJS dependency management techniques:

- **Forward References**: Uses `forwardRef()` to resolve circular dependencies between services
- **Decoupled Evaluation**: The evaluation system can operate independently while still integrating with the core services
- **Intelligent Fallbacks**: Services gracefully degrade when dependencies are initializing

```typescript
// Example of circular dependency resolution
constructor(
  @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
  private readonly modelUtilsService: ModelUtilsService,
  @Inject(forwardRef(() => ModelEvaluationService)) 
  private readonly modelEvaluationService?: ModelEvaluationService,
) {}
```

### TensorFlow.js Integration

The system integrates TensorFlow.js for real model training:

- **Dynamic Loading**: TensorFlow.js is loaded on-demand to minimize resource usage
- **Custom Tokenization**: Implements a specialized tokenizer for processing text inputs
- **Model Persistence**: Trained models are saved to disk for future use
- **Inference Pipeline**: Complete pipeline from tokenization to prediction

### Circuit Breaker Implementation

The system now implements the Circuit Breaker pattern for enhanced resilience:

- **Automatic Failure Detection**: Detects failing API providers and opens the circuit to prevent cascading failures
- **Graceful Fallbacks**: Automatically redirects requests to alternative providers when the primary fails
- **Self-Healing**: Circuit breakers automatically attempt to recover after a cooling-off period
- **Configuration by Service**: Different failure thresholds and timeouts for different service types

```typescript
// Example of circuit breaker configuration
const DEFAULT_CIRCUIT_BREAKER_CONFIGS: Record<string, CircuitBreakerConfig> = {
  'google-ai': {
    failureThreshold: 3,  // 3 consecutive failures open the circuit
    resetTimeout: 30000,  // 30 seconds before going to half-open state
    successThreshold: 2,  // 2 successes close the circuit again
    timeout: 10000,       // 10 second timeout for requests
    monitorInterval: 60000, // Monitor metrics every 60 seconds
    name: 'google-ai'
  }
}
```

### Anomaly Detection Framework

The system includes a sophisticated anomaly detection framework:

- **Multiple Detection Levels**: Configure sensitivity from LOW to HIGH
- **Content Analysis**: Detects logical inconsistencies, biases, and factual errors
- **Mitigation Strategies**: Provides suggestions to address detected anomalies
- **Real-time Monitoring**: Checks responses as they are generated

## Project Structure

Based on the source code, the project is organized as follows:

```
/src
├── /config             # Configuration settings
├── /types              # TypeScript type definitions  
├── /rag-kag            # Core RAG/KAG implementation
│   ├── /apis           # LLM API integrations
│   │   ├── google-ai.service.ts           # Google AI integration
│   │   ├── qwen-ai.service.ts             # Qwen AI integration
│   │   ├── deepseek-ai.service.ts         # DeepSeek integration
│   │   ├── house-model.service.ts         # Local model implementation
│   │   ├── model-training.service.ts      # Model training scheduler
│   │   ├── model-evaluation.service.ts    # Model evaluation framework
│   │   ├── model-utils.service.ts         # TensorFlow utility service
│   │   ├── tokenizer.service.ts           # Text tokenization service
│   │   └── api-provider-factory.service.ts  # API factory pattern
│   ├── /agents         # Agent implementations
│   ├── /controllers    # API controllers
│   ├── /debate         # RAG/KAG debate system
│   ├── /orchestrator   # Query orchestration
│   ├── /pools          # Agent pool implementations
│   │   ├── commercial-pool.service.ts     # Commercial domain agents
│   │   ├── marketing-pool.service.ts      # Marketing domain agents
│   │   ├── sectorial-pool.service.ts      # Sector-specific agents
│   │   ├── educational-pool.service.ts    # Educational content agents
│   │   └── pool-manager.service.ts        # Pool coordination
│   ├── /prompts        # Prompt templates
│   ├── /synthesis      # Response synthesis
│   └── /utils          # Utilities
├── /utils              # General utilities
└── /examples           # Example implementations
```

## Technologies Used

The project leverages the following technologies:

- **Backend Framework**: NestJS
- **Language**: TypeScript
- **API Documentation**: Swagger via @nestjs/swagger
- **LLM Integration**: Support for multiple providers
- **Scheduling**: @nestjs/schedule for periodic tasks
- **Model Management**: Automated training and distillation

## Development

### Testing

```bash
# Run unit tests
yarn test

# Run integration tests
yarn test:e2e

# Generate test coverage
yarn test:cov
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
