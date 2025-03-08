import sys
import os
from text_extractor import TextExtractor
from pinecone_client import PineconeClient
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Pinecone credentials from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
PINECONE_AWS_REGION = os.getenv("PINECONE_AWS_REGION", "us-east-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")

# Initialize Pinecone client at script level
pinecone_client = None

# Define the state schema for our graph
class GraphState(TypedDict):
    pdf_path: str
    text: str
    chunks: List[str]
    error: str
    metadata: Dict[str, Any]
    index_stats: Dict[str, Any]

# Function to extract text from PDF
def extract_text(state: GraphState) -> GraphState:
    try:
        extractor = TextExtractor(state["pdf_path"])
        text = extractor.extract_text()
        return {"text": text, "metadata": {"source": state["pdf_path"]}}
    except Exception as e:
        return {"error": f"Error extracting text: {str(e)}"}

# Function to split text into chunks using LangChain's text splitter
def split_text(state: GraphState) -> GraphState:
    try:
        text = state["text"]
        metadata = state.get("metadata", {})
        
        # Create a RecursiveCharacterTextSplitter
        # This is one of the most versatile splitters in LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a relatively small chunk size for demonstration
            chunk_size=500,
            chunk_overlap=100,  # Some overlap between chunks to maintain context
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Try to split on paragraphs first, then newlines, etc.
        )
        
        # Split the text into chunks
        chunks = text_splitter.split_text(text)
        
        # Add metadata to the result
        return {
            "chunks": chunks,
            "metadata": {
                **metadata,
                "num_chunks": len(chunks),
                "chunk_size": 500,
                "chunk_overlap": 100
            }
        }
    except Exception as e:
        return {"error": f"Error splitting text: {str(e)}"}

# Function to store chunks in Pinecone
def store_in_pinecone(state: GraphState) -> GraphState:
    try:
        global pinecone_client
        chunks = state["chunks"]
        metadata = state.get("metadata", {})
        source = metadata.get("source", "unknown")
        
        # Use the global Pinecone client
        index_stats = pinecone_client.store_chunks(chunks, source)
        
        return {"index_stats": index_stats}
    except Exception as e:
        return {"error": f"Error storing in Pinecone: {str(e)}"}

# Function to check for errors
def has_error(state: GraphState) -> str:
    if state.get("error"):
        return "error"
    return "continue"

def main():
    # Check if PDF path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python vectorize.py <path_to_pdf_file> [--reset-index]")
        sys.exit(1)
    
    # Get the PDF path from command-line arguments
    pdf_path = sys.argv[1]
    
    # Check if reset_index flag is provided (optional)
    reset_index = "--reset-index" in sys.argv
    
    try:
        # Check if Pinecone credentials are available
        if not PINECONE_API_KEY:
            print("Error: Missing PINECONE_API_KEY in .env file")
            sys.exit(1)
        
        if not PINECONE_INDEX_NAME:
            print("Error: Missing PINECONE_INDEX in .env file")
            sys.exit(1)
        
        # Initialize the global Pinecone client
        global pinecone_client
        pinecone_client = PineconeClient(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            aws_region=PINECONE_AWS_REGION,
            embedding_model=EMBEDDING_MODEL,
            reset_index=reset_index
        )
        
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes to the graph
        workflow.add_node("extract_text", extract_text)
        workflow.add_node("split_text", split_text)
        workflow.add_node("store_in_pinecone", store_in_pinecone)
        workflow.add_node("output", lambda x: x)  # Pass-through node for final state
        
        # Add edges to connect the nodes
        workflow.add_edge("extract_text", "split_text")
        workflow.add_edge("split_text", "store_in_pinecone")
        workflow.add_edge("store_in_pinecone", "output")
        
        # Add conditional edge for error handling
        workflow.add_conditional_edges(
            "extract_text",
            has_error,
            {
                "error": "output",
                "continue": "split_text"
            }
        )
        
        workflow.add_conditional_edges(
            "split_text",
            has_error,
            {
                "error": "output",
                "continue": "store_in_pinecone"
            }
        )
        
        workflow.add_conditional_edges(
            "store_in_pinecone",
            has_error,
            {
                "error": "output",
                "continue": "output"
            }
        )
        
        # Set the entry point
        workflow.set_entry_point("extract_text")
        
        # Compile the graph
        graph = workflow.compile()
        
        # Execute the graph with initial state
        result = graph.invoke({"pdf_path": pdf_path, "reset_index": reset_index})
        
        # Print results
        if result.get("error"):
            print(f"Error: {result['error']}")
            sys.exit(1)
        
        print(f"Successfully processed PDF: {pdf_path}")
        print(f"Extracted {len(result['text'])} characters of text")
        print(f"Split into {len(result['chunks'])} chunks")
        print(f"Chunk size: {result['metadata']['chunk_size']} characters")
        print(f"Chunk overlap: {result['metadata']['chunk_overlap']} characters")
        
        # Print Pinecone upload stats if available
        if result.get("index_stats"):
            print("\nPinecone Upload Statistics:")
            print(f"Index Name: {result['index_stats']['index_name']}")
            print(f"Vectors Uploaded: {result['index_stats']['vectors_uploaded']}")
            print(f"Source: {result['index_stats']['source']}")
            print(f"AWS Region: {result['index_stats']['aws_region']}")
            print(f"Model: {result['index_stats']['model_name']}")
            print(f"Reset Index: {result['index_stats']['reset_index']}")
        
        # Print first few chunks as a sample
        print("\nSample chunks:")
        for i, chunk in enumerate(result['chunks'][:3]):
            print(f"\nChunk {i+1}:")
            print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
        
        if len(result['chunks']) > 3:
            print(f"\n... and {len(result['chunks']) - 3} more chunks")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 