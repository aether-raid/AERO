import urllib.request as libreq
import xml.etree.ElementTree as ET

def format_search_string(input_string):
    """Convert string to arXiv search format handling slash-separated terms.
    
    Input: "deep learning/time series/forecasting/variable length"
    Output: 'all:%22deep+learning%22+AND+all:%22time+series%22+AND+all:forecasting+AND+all:%22variable+length%22'
    """
    # Split by forward slashes
    terms = input_string.strip().split('/')
    parts = []
    
    for term in terms:
        term = term.strip()
        if not term:
            continue
        
        # If term has spaces, treat it as a phrase (add quotes and encoding)
        if ' ' in term:
            # Replace spaces with + and add URL encoding for quotes
            formatted_term = term.replace(' ', '+')
            parts.append(f'all:%22{formatted_term}%22')
        else:
            # Single word, no quotes needed
            parts.append(f'all:{term}')
    
    # Join with AND
    return '+'.join(parts) if parts else ""


def explore_atom_elements(xml_data):
    """Explore and display all atom element names in the XML response."""
    root = ET.fromstring(xml_data)
    
    # Namespaces
    namespaces = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom',
        'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
    }
    
    print("🔍 ATOM ELEMENT EXPLORER")
    print("="*60)
    
    print("\n📊 FEED-LEVEL ELEMENTS:")
    feed_elements = {}
    for child in root:
        tag_name = child.tag
        # Remove namespace prefix for cleaner display
        clean_name = tag_name.split('}')[-1] if '}' in tag_name else tag_name
        namespace = tag_name.split('}')[0].strip('{') if '}' in tag_name else "no namespace"
        feed_elements[clean_name] = namespace
    
    for element, ns in sorted(feed_elements.items()):
        ns_short = ns.split('/')[-1] if '/' in ns else ns
        print(f"  • {element} ({ns_short})")
    
    print("\n📄 ENTRY-LEVEL ELEMENTS:")
    entry = root.find('atom:entry', namespaces)
    if entry is not None:
        entry_elements = {}
        for child in entry:
            tag_name = child.tag
            clean_name = tag_name.split('}')[-1] if '}' in tag_name else tag_name
            namespace = tag_name.split('}')[0].strip('{') if '}' in tag_name else "no namespace"
            entry_elements[clean_name] = namespace
        
        for element, ns in sorted(entry_elements.items()):
            ns_short = ns.split('/')[-1] if '/' in ns else ns
            print(f"  • {element} ({ns_short})")
    
    print("\n🏷️ AVAILABLE NAMESPACES:")
    print(f"  • atom: {namespaces['atom']}")
    print(f"  • arxiv: {namespaces['arxiv']}")
    print(f"  • opensearch: {namespaces['opensearch']}")
    
    return namespaces


if __name__ == "__main__":
    # Example usage
    search_string = "deep learning/time series forecasting/variable length"
    search_query = format_search_string(search_string)

    print(f"Original string: {search_string}")
    print(f"Formatted query: {search_query}")
    print("="*80)
    print("="*80)
    url = f"http://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results=5"

    print("🔍 SEARCHING: Deep learning time series variable length")
    print("="*80)
    print(f"Query: {search_query}")
    print("="*80)

    try:
        with libreq.urlopen(url) as response:
            xml_data = response.read()
        
        # First, explore the atom elements
        namespaces = explore_atom_elements(xml_data)
        
        # Parse XML
        root = ET.fromstring(xml_data)
        
        # Namespaces (from explorer)
        ns = namespaces
        
        # Get total results
        total_results_elem = root.find('opensearch:totalResults', ns)
        total_results = total_results_elem.text if total_results_elem is not None else "Unknown"
        
        print(f"Total papers found: {total_results}")
        print("="*80)
        
        # Get all paper entries
        entries = root.findall('atom:entry', ns)
        
        if entries:
            for i, entry in enumerate(entries, 1):
                # Extract basic info
                title = entry.find('atom:title', ns).text.strip() # type: ignore
                summary = entry.find('atom:summary', ns).text.strip() # type: ignore
                paper_id = entry.find('atom:id', ns).text.split('/')[-1] # type: ignore
                content = entry.find('atom:content', ns).text.strip() if entry.find('atom:content', ns) is not None else "No content" # type: ignore

                # Get authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns).text # type: ignore
                    authors.append(name)
                
                # Get published date
                published = entry.find('atom:published', ns).text[:10] if entry.find('atom:published', ns) is not None else "Unknown" # type: ignore
                
                # Print formatted output
                print(f"\n📄 PAPER #{i}")
                print("-" * 60)
                print(f"Title: {title}")
                print(f"ID: {paper_id}")
                print(f"Published: {published}")
                print(f"Authors: {', '.join(authors)}")
                print(f"Content:{content}")
                print("-" * 60)
                print(f"Abstract:\n{summary}")
                print("="*80)
        else:
            print("❌ No papers found")
            
    except Exception as e:
        print(f"❌ Error with search: {e}")
        