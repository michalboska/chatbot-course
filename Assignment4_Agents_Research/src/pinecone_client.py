import uuid
import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass
from pinecone import Pinecone, ServerlessSpec

@dataclass
class SearchResult:
    """Data class representing a search result from Pinecone."""
    score: float
    chunk_text: str
    source: str
    chunk_index: int
    total_chunks: int
    id: str

class PineconeClient:
    def __init__(self, api_key: str, index_name: str, aws_region: str = "us-west-2", embedding_model: str = "llama-text-embed-v2", reset_index: bool = False):
        """
        Initialize the Pinecone client with configuration parameters.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            aws_region: AWS region for the Pinecone index (default: us-west-2)
            embedding_model: Embedding model to use (default: llama-text-embed-v2)
            reset_index: If True, drop the index if it exists and recreate it (default: False)
        """
        self.api_key = api_key
        self.index_name = index_name
        self.aws_region = aws_region
        self.embedding_model = embedding_model
        self.reset_index = reset_index
        
        # Initialize Pinecone client
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        if not self.index_name:
            raise ValueError("Pinecone index name is required")
        
        self.pc = Pinecone(api_key=self.api_key)
        self._ensure_index_exists()
        
    def _ensure_index_exists(self):
        """Ensure that the Pinecone index exists, creating it if necessary."""
        index_exists = self.index_name in self.pc.list_indexes().names()
        
        # If reset_index is True and the index exists, delete it
        if self.reset_index and index_exists:
            self.pc.delete_index(self.index_name)
            index_exists = False
            
        # Create the index if it doesn't exist
        if not index_exists:
            # Create index using the specified embedding model
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud="aws",
                region=self.aws_region,
                embed={
                    "model":self.embedding_model,
                    "field_map":{"text": "chunk_text"}
                }
            )
    
    def _generate_chunk_id(self, chunk: str, source: str, index: int) -> str:
        """
        Generate a deterministic ID for a chunk using MD5 hash.
        
        Args:
            chunk: The text chunk
            source: Source of the chunk
            index: Index of the chunk
            
        Returns:
            A deterministic ID based on the chunk content and metadata
        """
        # Create a string that combines the chunk, source, and index
        content_to_hash = f"{source}_{index}_{chunk}"
        
        # Create MD5 hash
        md5_hash = hashlib.md5(content_to_hash.encode()).hexdigest()
        
        return md5_hash
    
    def store_chunks(self, chunks: List[str], source: str) -> Dict[str, Any]:
        """
        Store text chunks in Pinecone.
        
        Args:
            chunks: List of text chunks to store
            source: Source of the chunks (e.g., file path)
            
        Returns:
            Dictionary with statistics about the operation
        """
        # Get the index
        index = self.pc.Index(self.index_name)
        
        # Prepare vectors for upsert
        vectors = []
        for i, chunk in enumerate(chunks):
            # Generate a deterministic ID for each chunk using MD5
            vector_id = self._generate_chunk_id(chunk, source, i)
                        
            # Add vector to the list
            vectors.append({
                "id": vector_id,
                "chunk_text": chunk,
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
        
        # Upsert vectors to Pinecone in batches
        batch_size = 90  # Slightly below the 96 limit for safety
        total_vectors = len(vectors)
        vectors_uploaded = 0
        
        for i in range(0, total_vectors, batch_size):
            # Get the current batch
            batch = vectors[i:i + batch_size]
            
            # Upsert the batch
            index.upsert_records(namespace="", records=batch)
            
            # Update the count of uploaded vectors
            vectors_uploaded += len(batch)
            
            # Optional: Print progress
            print(f"Uploaded batch {i // batch_size + 1}/{(total_vectors + batch_size - 1) // batch_size}: {vectors_uploaded}/{total_vectors} vectors")
        
        # Return statistics about the operation
        return {
            "index_name": self.index_name,
            "vectors_uploaded": vectors_uploaded,
            "source": source,
            "aws_region": self.aws_region,
            "model_name": self.embedding_model,
            "reset_index": self.reset_index
        }
    
    def search_query(self, query: str, top_k: int = 5, source: str = None) -> List[SearchResult]:
        """
        Search for top matches against the Pinecone index.
        
        Args:
            query: The query text to search for
            top_k: Number of top results to return (default: 5)
            source: Optional source to filter results by (default: None)
            
        Returns:
            List of SearchResult objects containing the top matching chunks and their metadata
        """
        # Get the index
        index = self.pc.Index(self.index_name)
        
        # Prepare filter if source is provided
        filter_dict = None
        if source is not None:
            filter_dict = {"source": {"$eq": source}}
        
        # Query the index
        query_response = index.search_records(
            namespace="",
            query={
                "inputs": {"text": query},
                "top_k": top_k,
                "filter": filter_dict or {}
            },
            fields=["chunk_text", "source", "chunk_index", "total_chunks", "id"]
        )

        # Process and return the results
        results = []

        for query_hit in query_response.result.hits:
            results.append(SearchResult(
                score=query_hit._score,
                chunk_text=query_hit.fields.get("chunk_text", ""),
                source=query_hit.fields.get("source", "unknown"),
                chunk_index=query_hit.fields.get("chunk_index", 0),
                total_chunks=query_hit.fields.get("total_chunks", 0),
                id=query_hit.id
            ))
        
        return results 