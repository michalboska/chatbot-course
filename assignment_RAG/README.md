# PDF Vectorization and RAG System

This repository contains tools for vectorizing PDF documents and implementing a Retrieval-Augmented Generation (RAG) system using Pinecone and OpenAI.

## Setup

### Prerequisites

- Python 3.8+
- Pinecone account (for vector database)
- OpenAI API key (for embeddings and chat completions)

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on the provided `.env.example`:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file and add your API keys:
   ```
   PINECONE_API_KEY=your_actual_pinecone_api_key
   PINECONE_INDEX=your_pinecone_index_name
   PINECONE_AWS_REGION=your_preferred_region
   OPENAI_API_KEY=your_actual_openai_api_key
   ```

## Vectorizing PDF Documents

The `vectorize.py` script processes PDF files, extracts text, splits it into chunks, and stores the chunks in a Pinecone vector database.

### Usage

```bash
python vectorize.py <path_to_pdf_file> [--reset-index]
```

#### Parameters:

- `<path_to_pdf_file>`: Path to the PDF file you want to process
- `--reset-index` (optional): If provided, the existing Pinecone index will be deleted before storing new vectors

### Example

```bash
python vectorize.py documents/research_paper.pdf
```

To reset the index before processing:

```bash
python vectorize.py documents/research_paper.pdf --reset-index
```

### Process Flow

1. **Text Extraction**: The script extracts text from the PDF file
2. **Text Splitting**: The extracted text is split into smaller chunks (default: 500 characters with 100 character overlap)
3. **Vector Storage**: The chunks are embedded and stored in Pinecone


## Using the RAG Chat System

After vectorizing your documents, you can use the chat system to query the knowledge base:

```bash
python chat.py
```

This will start an interactive chat session where you can ask questions about the documents you've vectorized.

## Troubleshooting

- **Missing API Keys**: Ensure all required API keys are properly set in your `.env` file
- **PDF Processing Errors**: Make sure the PDF is not password-protected and is readable
- **Batch Size Errors**: For large PDFs, the system automatically handles batching to avoid Pinecone limits
