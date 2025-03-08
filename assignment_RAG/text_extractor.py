import os
import sys
import pypdf

class TextExtractor:
    """
    A class for extracting text from PDF files.
    """
    
    def __init__(self, file_path):
        """
        Initialize the TextExtractor with a file path.
        
        Args:
            file_path (str): Path to the PDF file
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a PDF
        """
        # Validate that the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        
        # Validate that the file is a PDF (basic check by extension)
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"The file '{file_path}' does not appear to be a PDF file.")
        
        self.file_path = file_path
    
    def extract_text(self):
        """
        Extract text from the PDF file.
        
        Returns:
            str: Extracted text from the PDF
            
        Raises:
            Exception: If there's an error during text extraction
        """
        try:
            # Open the PDF file
            with open(self.file_path, 'rb') as file:
                # Create a PDF reader object
                pdf_reader = pypdf.PdfReader(file)
                
                # Initialize an empty string to store the text
                text = ""
                
                # Iterate through each page and extract text
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                    
                return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {e}")

# Add this to make the class available when importing
__all__ = ['TextExtractor'] 