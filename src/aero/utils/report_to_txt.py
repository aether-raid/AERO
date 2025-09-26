import requests
import feedparser
import PyPDF2
from io import BytesIO
import re
from pathlib import Path

def sanitize_filename(filename):
    """Sanitize filename by removing invalid characters."""
    # Remove invalid characters for Windows/Unix filenames
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove excessive whitespace and dots
    filename = re.sub(r'\s+', ' ', filename).strip()
    filename = filename.replace('..', '.')
    # Limit length
    return filename[:100] if len(filename) > 100 else filename

def create_paper_directory(paper_title, paper_id, base_dir="downloaded_papers"):
    """Create directory structure for storing paper files.
    
    Args:
        paper_title: Title of the paper
        paper_id: ArXiv ID of the paper
        base_dir: Base directory for storing papers
        
    Returns:
        tuple: (paper_filename, pdf_dir_path, txt_dir_path)
    """
    # Sanitize the title for use as folder name
    safe_title = sanitize_filename(paper_title)
    safe_id = sanitize_filename(paper_id)
    
    # Create filename: "id_title"
    filename = f"{safe_id}_{safe_title}"
    
    # Create top-level directories
    base_path = Path(base_dir)
    pdf_dir = base_path / "pdf-papers"
    txt_dir = base_path / "txt-papers"
    
    # Create directories
    pdf_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created directories for papers: pdf-papers/ and txt-papers/")
    
    return filename, str(pdf_dir), str(txt_dir)

def extract_pdf_text(pdf_url, save_files=False, paper_title=None, paper_id=None):
    """
    Download and extract text from a PDF URL.
    
    Args:
        pdf_url (str): URL of the PDF to download
        save_files (bool): Whether to save PDF and text files
        paper_title (str): Title of the paper (for file naming)
        paper_id (str): ID of the paper (for file naming)
    
    Returns:
        str: Extracted text content
    """
    try:
       # print(f"Fetching PDF from: {pdf_url}")
        
        # Download the PDF
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        #print(f"✅ PDF downloaded ({len(response.content):,} bytes)")
        save=False
        # Save PDF file if requested and metadata is available
        pdf_path = None
        if save_files and paper_title and paper_id and save:
            try:
                filename, pdf_dir, txt_dir = create_paper_directory(paper_title, paper_id)
                pdf_filename = f"{filename}.pdf"
                pdf_path = Path(pdf_dir) / pdf_filename
                
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                #print(f"💾 PDF saved to: {pdf_path}")
            except Exception as e:
                print(f"Failed to save PDF: {e}")
                
        
        #print("📄 Extracting text content...")
        
        # Create a BytesIO object from the PDF content
        pdf_file = BytesIO(response.content)
        
        # Read PDF with PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text_content = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text_content += f"\n--- PAGE {page_num + 1} ---\n"
            text_content += page_text
            text_content += "\n" + "="*50 + "\n"
        
        #print(f"✅ Text extracted successfully ({len(pdf_reader.pages)} pages)")
        
        # Save text file if requested and metadata is available
        
        if save_files and paper_title and paper_id and text_content and save:
            try:
                if 'txt_dir' not in locals():
                    filename, pdf_dir, txt_dir = create_paper_directory(paper_title, paper_id)
                
                txt_filename = f"{filename}.txt"
                txt_path = Path(txt_dir) / txt_filename
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                print(f"💾 Text saved to: {txt_path}")
            except Exception as e:
                print(f"⚠️ Failed to save text file: {e}")
        
        return text_content
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return None
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def get_arxiv_paper_details(paper_id):
    """Get paper details from ArXiv API."""
    url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        feed = feedparser.parse(response.content)
        if feed.entries:
            entry = feed.entries[0]
            return {
                'title': entry.title,
                'authors': ', '.join([author.name for author in entry.authors]),
                'published': entry.published,
                'summary': entry.summary
            }
    return None

# Example usage
if __name__ == "__main__":
    # Example ArXiv paper ID
    paper_id = "2111.00715v1"
    
    # Get paper details
    details = get_arxiv_paper_details(paper_id)
    if details:
        print("="*60)
        print("📋 PAPER DETAILS")
        print("-"*60)
        print(f"Title: {details['title']}")
        print(f"Authors: {details['authors']}")
        print(f"Published: {details['published']}")
        print("="*60)
        
        # Construct PDF URL
        pdf_url = f"http://arxiv.org/pdf/{paper_id}"
        
        # Extract text and save files
        text_content = extract_pdf_text(pdf_url, save_files=True, 
                                      paper_title=details['title'], 
                                      paper_id=paper_id)
        
        if text_content:
            print("="*60)
            print("📄 PDF TEXT CONTENT (First 500 characters)")
            print("="*60)
            print(text_content[:500] + "..." if len(text_content) > 500 else text_content)
    else:
        print("❌ Could not fetch paper details")