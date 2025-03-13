import os
import sys
from typing import List, Dict, Any, TypedDict, Optional
from dotenv import load_dotenv
from pinecone_client import PineconeClient, SearchResult
from serp_client import ScholarResult, SerpClient
from openai import OpenAI
from langgraph.graph import StateGraph, END
import traceback
import webbrowser

# Load environment variables from .env file
load_dotenv()

# Get Pinecone credentials from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
PINECONE_AWS_REGION = os.getenv("PINECONE_AWS_REGION", "us-east-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Get SERP API key
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Initialize clients at script level
pinecone_client = None
openai_client = None
serp_client = None

# Define the state schema for our LangGraph
class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    has_research_data: bool
    context: Optional[List[SearchResult]]
    current_query: Optional[str]
    current_document: Optional[str]
    web_results: List[ScholarResult]

# Function to initialize OpenAI client
def init_openai_client():
    global openai_client
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is required")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return openai_client

# Function to initialize SERP client
def init_serp_client():
    global serp_client
    if not SERP_API_KEY:
        raise ValueError("SERP API key is required.")
    serp_client = SerpClient(api_key=SERP_API_KEY)
    return serp_client

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
        "context": search_results,
        "has_research_data": True,
    }


# Function to retrieve web search results
def retrieve_web_results(state: ChatState) -> ChatState:
    """Retrieve relevant information from web search based on the current query."""
    messages = state["messages"]
    # if no research data, return the state unchanged
    if not state["has_research_data"]:
        messages.append({"role": "assistant", "content": "I don't have any data to search the web for. Please ask me a question about the relevant topic first."})
        return {
            **state,
            "messages": messages
        }
    
    previous_messages_without_system = [msg for msg in messages if msg["role"] != "system"]
    
    search_query_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """
            You are an expert in researching information on the Internet using Google.
            You will be given a chat history of a previous conversation between a user and an assistant. 
            Your task is to summarize the conversation and create a search query to search the web for the information.
            The search query should be concise and to the point, and should be no more than 30 words. Only input the search query, nothing else.
            """},
        ] + previous_messages_without_system + [
            {"role": "user", "content": "Please create a search query to search the web for the information."}
        ],
        temperature=0,
        max_tokens=1000
    )
    search_query = search_query_response.choices[0].message.content.strip().replace("\"", "")    
    # Search for relevant information on the web
    web_results = serp_client.scholar_search(query=search_query, num_results=3)

    message = {}
    if len(web_results) > 0:
        formatted_web_results = [f"{i+1}. {result.title} - {result.link}" for i, result in enumerate(web_results)]
        formatted_results_str = "\n".join(formatted_web_results)
        message = {
            "role": "assistant",
            "content": f"I found the following sources for your search query: {search_query}\n\n{formatted_results_str}\n\nI can open these sources in the web browser for you if you want."
        }
        messages.append(message)
    else:
        message = {
            "role": "assistant",
            "content": f"I didn't find any sources for your search query: {search_query}"
        }
        messages.append(message)    
    
    # Update the state with the retrieved web results
    return {
        **state,
        "web_results": web_results,
        "messages": messages,
    }

def open_sources(state: ChatState) -> ChatState:
    """Open the found sources in the web browser."""
    messages = state["messages"]
    web_results = state["web_results"]
    if len(web_results) == 0:
        messages.append({"role": "assistant", "content": "I don't have any sources to open. Please search for information first."})
        return {
            **state,
            "messages": messages
        }   
    for result in web_results:
        webbrowser.open(result.link)
    return {
        **state,
        "messages": messages.append({"role": "assistant", "content": f"I've opened {len(web_results)} source(s) as tabs in your browser."})
    }


# Function to generate a response using GPT-4o
def generate_response(state: ChatState) -> ChatState:
    """Generate a response using GPT-4o based on the conversation history and retrieved context."""
    # Prepare the messages for the OpenAI API
    messages = []
    
    # Add system message with instructions
    
    
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
        "current_query": None,  # Reset the current query
    }

# Function to determine the intent of the user query
def determine_intent(state: ChatState) -> ChatState:
    """Determine the intent of the user query and classify it into predefined groups."""
    query = state["current_query"]
    if not query:
        # If no query is provided, return the state unchanged
        return state
    
    # Use OpenAI to classify the intent
    messages = [
        {"role": "system", "content": """
        You are an intent classifier for an architecture and psychology research assistant.
        Classify the user query into one of the following categories:
        - research_question: Questions about research studies, methodologies, or findings
        - web_search: Requests to search for information online or get latest information
        - general_question: General questions that don't fit other categories
        - open_sources: Requests to open found sources and references in the web browser
        - greeting: Greetings or salutations
        - farewell: Goodbyes or closing remarks
        - help: Requests for help or instructions on using the system
        
        Respond with ONLY the category name, nothing else.
        """},
        {"role": "user", "content": query}
    ]
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=20
    )
    
    # Extract the intent
    intent = response.choices[0].message.content.strip().lower()
    
    
    # Update the state with the detected intent
    return {
        **state,
        "intent": intent
    }

# Function to generate help information for the user
def print_help(state: ChatState) -> ChatState:
    """Generate a helpful response when the user asks for help with the bot."""
    # Create a new messages array starting with the system message
    user_message = """
    I need help with the bot. Using your system message, please explain how to use the bot.
    
    Make the response friendly, concise, and informative.
    """
    
    # Create messages array with system message first, then all existing messages
    updated_messages = state["messages"] + [{"role": "user", "content": user_message}]    
    # Generate response using OpenAI API
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=updated_messages,
        temperature=0.0,
        max_tokens=500
    )
    
    # Extract the assistant's message
    help_message = response.choices[0].message.content
    
    # Update the state with the new message
    updated_messages = state["messages"] + [{"role": "assistant", "content": help_message}]
    
    return {
        **state,
        "messages": updated_messages,
        "current_query": None,  # Reset the current query
        "web_results": None,    # Reset the web results
    }

# Function to create the LangGraph workflow
def create_rag_workflow():
    # Create a new graph
    workflow = StateGraph(ChatState)
    
    # Add nodes to the graph
    workflow.add_node("determine_intent", determine_intent)
    workflow.add_node("retrieve_RAG", retrieve_context)
    workflow.add_node("generate_LLM", generate_response)
    workflow.add_node("retrieve_web", retrieve_web_results)
    workflow.add_node("open_sources", open_sources)
    workflow.add_node("print_help", print_help)
    # Define the conditional edge based on intent
    def route_by_intent(state: ChatState) -> str:
        intent = state.get("intent", "general_question")
        
        if intent == "web_search":
            return "retrieve_web"
        elif intent == "open_sources":
            return "open_sources"
        elif intent == "research_question":
            return "retrieve_RAG"
        elif intent == "help":
            return "print_help"
        elif intent in ["greeting", "farewell"]:
            return "generate_LLM"
        else:
            # Default path
            return "retrieve_RAG"
    
    # Define the edges with conditional routing
    workflow.add_conditional_edges(
        "determine_intent",
        route_by_intent,
        {
            "retrieve_RAG": "retrieve_RAG",
            "retrieve_web": "retrieve_web",
            "open_sources": "open_sources",
            "generate_LLM": "generate_LLM",
            "print_help": "print_help"
        }
    )
    
    # Add remaining edges
    workflow.add_edge("retrieve_RAG", "generate_LLM")
    workflow.add_edge("retrieve_web", END)
    workflow.add_edge("open_sources", END)
    workflow.add_edge("print_help", END)
    workflow.add_edge("generate_LLM", END)
    
    # Set the entry point
    workflow.set_entry_point("determine_intent")
    
    # Compile the graph
    compiled_workflow = workflow.compile()
    

    return compiled_workflow



# Function to process a user query
def process_query(query: str, conversation_state: ChatState) -> ChatState:
    """Process a user query through the RAG workflow."""
    global rag_workflow
    
    try:
        # Add the user query to the conversation history
        conversation_state["messages"].append({"role": "user", "content": query})
        conversation_state["current_query"] = query
        
        # Run the workflow with debug enabled to see the flow
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

    system_message = """
    You are a research expert in following topics:
        - Architecture and its effect on human emotions
        - Effects of architecture on psychology
        - Research methods
        - Summarizing existing research and articles

    You don't have expertise in any other domains not related to architecture or
    research. Refuse to answer questions unrelated to these topics. If an answer to the question is not found in the conversation or context, say you
    don't have an answer and don't try to make up your own.

    User can ask you to find sources for a given topic. If they do, you should search the web for the sources and return them to the user.
    User can ask you to open the found sources in the web browser. If they do, you should open the sources in the web browser.

    Sources will be provided in the prompt in the form similar to this:
    [Source 1: <source_pdf_name.pdf>]

    When answering questions, always cite your sources, using the following format:
    "I found this information in <source_pdf_name.pdf>"
    """
    # Initialize the conversation state
    conversation_state: ChatState = {
        "messages": [{"role": "system", "content": system_message}],
        "context": None,
        "current_query": None,
        "current_document": None,
        "web_results": [],
        "has_research_data": False,
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
    global pinecone_client, openai_client, serp_client, rag_workflow
    
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
    
    # Check if SERP API key is available
    if not SERP_API_KEY:
        print("Error: SERP API key not found. Please check your .env file.")
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
        
        # Initialize SERP client
        init_serp_client()
        print("SERP client initialized")
        
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