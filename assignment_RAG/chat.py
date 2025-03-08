import os
import sys
from typing import List, Dict, Any, TypedDict, Optional
from dotenv import load_dotenv
from pinecone_client import PineconeClient, SearchResult
from openai import OpenAI
from langgraph.graph import StateGraph, END
import traceback

# Load environment variables from .env file
load_dotenv()

# Get Pinecone credentials from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
PINECONE_AWS_REGION = os.getenv("PINECONE_AWS_REGION", "us-east-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients at script level
pinecone_client = None
openai_client = None

# Define the state schema for our LangGraph
class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    context: Optional[List[SearchResult]]
    current_query: Optional[str]

# Function to initialize OpenAI client
def init_openai_client():
    global openai_client
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is required")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return openai_client

# Function to retrieve context from Pinecone
def retrieve_context(state: ChatState) -> ChatState:
    """Retrieve relevant context from Pinecone based on the current query."""
    query = state["current_query"]
    if not query:
        # If no query is provided, return the state unchanged
        return state
    
    # Search for relevant chunks in Pinecone
    search_results = pinecone_client.search_query(query=query, top_k=3)
    
    # Update the state with the retrieved context
    return {
        **state,
        "context": search_results
    }

# Function to generate a response using GPT-4o
def generate_response(state: ChatState) -> ChatState:
    """Generate a response using GPT-4o based on the conversation history and retrieved context."""
    # Prepare the messages for the OpenAI API
    messages = []
    
    # Add system message with instructions
    system_message = """
    You are a research expert in following topics:
        - Architecture and its effect on human emotions
        - Effects of architecture on psychology
        - Research methods
        - Summarizing existing research and articles

    You don't have expertise in any other domains not related to architecture or
    research. Refuse to answer questions unrelated to these topics. If an answer to the question is not found in the conversation or context, say you
    don't have an answer and don't try to make up your own.

    When answering questions, always cite your sources. Sources will be provided in the prompt in the form similar to this:
      [Source 1: <source_pdf_name.pdf>]
    When citing sources, use the following format:
    "I found this information in <source_pdf_name.pdf>"
    """
    messages.append({"role": "system", "content": system_message})
    
    # Add conversation history
    for msg in state["messages"]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add context from RAG if available
    if state.get("context"):
        context_str = "Here is some relevant context to help answer the question:\n\n"
        for i, result in enumerate(state["context"]):
            context_str += f"[Source {i+1}: {result.source}]\n{result.chunk_text}\n\n"
        
        messages.append({"role": "system", "content": context_str})
    
    # Generate response using OpenAI API
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=1000
    )
    
    # Extract the assistant's message
    assistant_message = response.choices[0].message.content
    
    # Update the state with the new message
    updated_messages = state["messages"] + [{"role": "assistant", "content": assistant_message}]
    
    return {
        **state,
        "messages": updated_messages,
        "current_query": None  # Reset the current query
    }

# Function to create the LangGraph workflow
def create_rag_workflow():
    # Create a new graph
    workflow = StateGraph(ChatState)
    
    # Add nodes to the graph
    workflow.add_node("retrieve_RAG", retrieve_context)
    workflow.add_node("generate_LLM", generate_response)
    
    # Define the edges
    workflow.add_edge("retrieve_RAG", "generate_LLM")
    workflow.add_edge("generate_LLM", END)
    
    # Set the entry point
    workflow.set_entry_point("retrieve_RAG")
    
    # Compile the graph
    return workflow.compile()

# Function to process a user query
def process_query(query: str, conversation_state: ChatState) -> ChatState:
    """Process a user query through the RAG workflow."""
    global rag_workflow
    
    try:
        # Add the user query to the conversation history
        conversation_state["messages"].append({"role": "user", "content": query})
        conversation_state["current_query"] = query
        
        # Run the workflow
        final_state = rag_workflow.invoke(conversation_state, debug=False)
        
        return final_state
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Return the original state with an error message
        error_message = f"I encountered an error: {str(e)}"
        conversation_state["messages"].append({"role": "assistant", "content": error_message})
        return conversation_state

# Function to start a chat session
def start_chat_session():
    """Start an interactive chat session with the RAG system."""
    # Initialize the conversation state
    conversation_state: ChatState = {
        "messages": [],
        "context": None,
        "current_query": None
    }
    
    print("RAG Chat System initialized. Type 'exit' to quit.")
    print("Ask a question about the documents in the Pinecone index:")
    
    while True:
        # Get user input
        user_input = input("\nYou:\n")
        
        # Check if the user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        # Process the query
        conversation_state = process_query(user_input, conversation_state)
        
        # Display the assistant's response
        assistant_message = conversation_state["messages"][-1]["content"]
        print(f"\nAssistant: {assistant_message}")

def main():
    """
    Main function to run the chat application
    """
    # Initialize Pinecone client
    global pinecone_client, openai_client, rag_workflow
    
    # Check if Pinecone API key is available
    if not PINECONE_API_KEY:
        print("Error: Pinecone API key not found. Please check your .env file.")
        sys.exit(1)
    
    # Check if Pinecone index name is available
    if not PINECONE_INDEX_NAME:
        print("Error: Pinecone index name not found. Please check your .env file.")
        sys.exit(1)
    
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        print("Error: OpenAI API key not found. Please check your .env file.")
        sys.exit(1)
    
    try:
        # Initialize Pinecone client
        pinecone_client = PineconeClient(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            aws_region=PINECONE_AWS_REGION,
            embedding_model=EMBEDDING_MODEL
        )
        print(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
        
        # Initialize OpenAI client
        init_openai_client()
        print("Successfully connected to OpenAI API")
        
        # Create the RAG workflow once
        rag_workflow = create_rag_workflow()
        print("RAG workflow initialized")
        
        # Start the chat session
        start_chat_session()
        
    except Exception as e:
        print(f"Error initializing clients: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 