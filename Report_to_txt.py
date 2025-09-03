import requests
import feedparser
import os
from urllib.parse import urlparse
import PyPDF2
from io import BytesIO

def extract_pdf_text(pdf_url):
    """Download PDF and extract text content."""
    try:
        print(f"Fetching PDF from: {pdf_url}")
        
        # Get the PDF content
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        print(f"✅ PDF downloaded ({len(response.content):,} bytes)")
        print("📄 Extracting text content...")
        
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
        
        print(f"✅ Text extracted successfully ({len(pdf_reader.pages)} pages)")
        return text_content
        
    except Exception as e:
        print(f"❌ Error extracting PDF text: {e}")
        return None

# Example: Get metadata for a paper by ID
arxiv_id = "2406.05088v1"
url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

# Create downloads directory if it doesn't exist
if not os.path.exists('downloads'):
    os.makedirs('downloads')

response = requests.get(url)
feed = feedparser.parse(response.text)
entry = feed.entries[0]

print("="*60)
print("📋 PAPER DETAILS")
print("="*60)
print("Title:", entry.title)
print("Authors:", ', '.join([author.name for author in entry.authors]))
print("Published:", entry.published)
print("="*60)
print("Abstract:", entry.summary[:200] + "...")
print("="*60)

# Find PDF link
pdf_link = None
for link in entry.links:
    if link.type == 'application/pdf':
        pdf_link = link.href
        break

if pdf_link:
    print("PDF Link:", pdf_link)
    
    # Extract text from PDF
    pdf_text = extract_pdf_text(pdf_link)
    if pdf_text:
        print("\n" + "="*60)
        print("📄 PDF TEXT CONTENT")
        print("="*60)
        print(pdf_text[:2000] + "..." if len(pdf_text) > 2000 else pdf_text)
        
        # Optionally save text to file
        with open(f"downloads/{arxiv_id}_text.txt", 'w', encoding='utf-8') as f:
            f.write(pdf_text)
        print(f"\n� Full text saved as: downloads/{arxiv_id}_text.txt")
else:
    print("❌ No PDF link found")
