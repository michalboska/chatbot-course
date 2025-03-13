# LangGraph Workflow Visualization

This diagram visualizes the workflow of the Research Assistant system implemented using LangGraph.

```mermaid
graph TD
    %% Define nodes
    Start([Start]) --> determine_intent[determine_intent]
    determine_intent --> Intent{Intent?}
    
    %% Define conditional paths based on intent
    Intent -->|web_search| retrieve_web[retrieve_web]
    Intent -->|research_question| retrieve_RAG[retrieve_RAG]
    Intent -->|open_sources| open_sources[open_sources]
    Intent -->|help| print_help[print_help]
    Intent -->|greeting/farewell| generate_LLM[generate_LLM]
    Intent -->|general_question/default| retrieve_RAG
    
    %% Define remaining edges
    retrieve_RAG --> generate_LLM
    retrieve_web --> End([End])
    open_sources --> End
    print_help --> End
    generate_LLM --> End
    
    %% Styling with improved colors and readability
    classDef process fill:#4682B4,stroke:#333,stroke-width:1px,color:white; %% Steel Blue with white text
    classDef decision fill:#F4A460,stroke:#333,stroke-width:2px,color:#000000,font-weight:bold; %% Sandy Brown with bold black text
    classDef endpoint fill:#B0C4DE,stroke:#333,stroke-width:2px,color:black,stroke-dasharray: 5 5; %% Light Steel Blue with black text
    
    %% Apply classes to nodes
    class determine_intent,retrieve_RAG,retrieve_web,open_sources,print_help,generate_LLM process;
    class Intent decision;
    class Start,End endpoint;
```

## Node Descriptions

### 1. determine_intent
- **Purpose**: Analyzes the user query to determine its intent
- **Function**: Uses OpenAI to classify the query into predefined categories
- **Input**: User query
- **Output**: State with detected intent

### 2. retrieve_RAG
- **Purpose**: Retrieves relevant context from the Pinecone vector database
- **Function**: Searches for chunks of text related to the user's query
- **Input**: User query
- **Output**: State with retrieved context and has_research_data set to true

### 3. retrieve_web
- **Purpose**: Searches for academic papers related to the conversation
- **Function**: Uses the SERP API to search Google Scholar
- **Input**: Conversation history
- **Output**: State with web search results and assistant message

### 4. open_sources
- **Purpose**: Opens the found sources in the web browser
- **Function**: Uses the webbrowser module to open each link
- **Input**: Web results from previous searches
- **Output**: State with confirmation message

### 5. print_help
- **Purpose**: Generates a helpful response explaining how to use the bot
- **Function**: Uses OpenAI to create a help message based on the system message
- **Input**: Conversation history
- **Output**: State with help message

### 6. generate_LLM
- **Purpose**: Generates a response using GPT-4o
- **Function**: Uses OpenAI to create a response based on context and conversation
- **Input**: Conversation history and retrieved context
- **Output**: State with assistant's response

## Conditional Routing

The workflow routes the query based on the detected intent:

- **web_search**: Routes to `retrieve_web` to search for academic papers
- **research_question**: Routes to `retrieve_RAG` to get information from the vector database
- **open_sources**: Routes to `open_sources` to open found sources in the browser
- **help**: Routes to `print_help` to generate a help message
- **greeting/farewell**: Routes directly to `generate_LLM` to generate a simple response
- **general_question/default**: Routes to `retrieve_RAG` as the default path

## Flow Paths

1. **Research Question Path**: determine_intent → retrieve_RAG → generate_LLM → End
2. **Web Search Path**: determine_intent → retrieve_web → End
3. **Open Sources Path**: determine_intent → open_sources → End
4. **Help Path**: determine_intent → print_help → End
5. **Greeting/Farewell Path**: determine_intent → generate_LLM → End
6. **Default Path**: determine_intent → retrieve_RAG → generate_LLM → End
