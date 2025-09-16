from tavily import TavilyClient

tavily_client = TavilyClient(api_key="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")
search_response = tavily_client.search("Who is Leo Messi?")

context = []

context.append({
    "sources": [
        { "url": result["url"], "title": result["title"] } for result in search_response["results"]
    ]
})

extracted_results = []

for topic in context:
  extract_response = tavily_client.extract([source["url"] for source in topic["sources"]])

  for extracted_result in extract_response["results"]:
    for source in topic["sources"]:
      if source["url"] == extracted_result["url"]:
        source["content"] = extracted_result["raw_content"]

  for extracted_result in extract_response["failed_results"]:
    for source in topic["sources"]:
      if source["url"] == extracted_result["url"]:
        topic["sources"].remove(source)

  extracted_results.append(topic)

# Print results in a structured format
print("=" * 80)
print("SEARCH RESULTS FOR: Who is Leo Messi?")
print("=" * 80)

for i, topic in enumerate(extracted_results, 1):
    print(f"\nTOPIC {i}:")
    print("-" * 40)
    
    for j, source in enumerate(topic["sources"], 1):
        print(f"\nSOURCE {j}:")
        print(f"Title: {source['title']}")
        print(f"URL: {source['url']}")
        print(f"Content Length: {len(source.get('content', ''))} characters")
        print("-" * 60)
        
        # Print first 500 characters of content as preview
        content = source.get('content', 'No content available')
        if len(content) > 500:
            print("CONTENT PREVIEW:")
            print(content[:500] + "...")
            print(f"\n[Content truncated - showing first 500 of {len(content)} characters]")
        else:
            print("FULL CONTENT:")
            print(content)
        
        print("=" * 60)

print(f"\nSUMMARY:")
print(f"Total topics processed: {len(extracted_results)}")
total_sources = sum(len(topic['sources']) for topic in extracted_results)
print(f"Total sources found: {total_sources}")
print("=" * 80)