"""
Integration example: Using the enhanced Word formatter with ML research results.
This script demonstrates how to convert ML research pipeline output into 
professionally formatted Word documents.
"""

from word_formatter import WordFormatter
import json
import os
from datetime import datetime

def create_ml_research_report(research_results: dict, output_filename: str = None) -> str:
    """
    Convert ML research results into a professionally formatted Word document.
    
    Args:
        research_results: Dictionary containing ML research pipeline output
        output_filename: Optional custom filename for the output document
    
    Returns:
        str: Path to the created Word document
    """
    
    # Initialize formatter
    formatter = WordFormatter()
    
    # Extract key information from research results
    query = research_results.get('query', 'ML Research Analysis')
    timestamp = research_results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    recommendations = research_results.get('model_suggestions', '')
    
    # Create document title
    formatter.add_title(
        title=f"ML Research Report: {query}",
        subtitle=f"Generated on {timestamp}",
        add_date=False
    )
    
    # Add executive summary if available
    if 'executive_summary' in research_results:
        formatter.add_heading("Executive Summary", level=1)
        formatter.add_formatted_paragraph(research_results['executive_summary'])
        formatter.add_separator()
    
    # Add research context
    if 'research_context' in research_results:
        formatter.add_heading("Research Context", level=1)
        formatter.add_formatted_paragraph(research_results['research_context'])
        formatter.add_separator()
    
    # Add model recommendations (main content)
    if recommendations:
        formatter.add_heading("Model Recommendations", level=1)
        formatter.format_ml_text_recommendations(recommendations)
    
    # Add methodology if available
    if 'methodology' in research_results:
        formatter.add_page_break()
        formatter.add_heading("Methodology", level=1)
        formatter.add_formatted_paragraph(research_results['methodology'])
    
    # Add references/papers used
    if 'papers_analyzed' in research_results:
        formatter.add_heading("Papers Analyzed", level=2)
        papers = research_results['papers_analyzed']
        
        if isinstance(papers, list):
            paper_data = []
            for i, paper in enumerate(papers[:10], 1):  # Limit to top 10
                if isinstance(paper, dict):
                    title = paper.get('title', 'Unknown Title')
                    authors = paper.get('authors', 'Unknown Authors')
                    year = paper.get('year', 'N/A')
                    paper_data.append([str(i), title[:60] + "..." if len(title) > 60 else title, 
                                     authors[:40] + "..." if len(authors) > 40 else authors, str(year)])
            
            if paper_data:
                formatter.add_table(
                    paper_data,
                    headers=["#", "Title", "Authors", "Year"],
                    title="Key Papers Referenced"
                )
    
    # Add technical details if available
    if 'technical_analysis' in research_results:
        formatter.add_heading("Technical Analysis", level=2)
        formatter.add_formatted_paragraph(research_results['technical_analysis'])
    
    # Generate filename if not provided
    if not output_filename:
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"ml_research_report_{safe_query}_{timestamp_str}.docx"
    
    # Ensure .docx extension
    if not output_filename.endswith('.docx'):
        output_filename += '.docx'
    
    # Save document
    output_path = os.path.join(os.getcwd(), output_filename)
    formatter.save(output_path)
    
    return output_path

def demo_integration():
    """Demonstrate integration with sample ML research results."""
    
    # Sample research results (simulating ML pipeline output)
    sample_results = {
        "query": "Real-time Object Detection for Autonomous Vehicles",
        "timestamp": "2024-01-15 14:30:00",
        "executive_summary": "This analysis evaluates **state-of-the-art object detection models** for autonomous vehicle applications, focusing on *real-time performance* and accuracy trade-offs. The study analyzed 15 recent papers and identified 3 optimal model architectures.",
        "research_context": "Autonomous vehicles require object detection systems capable of processing **30+ FPS** while maintaining high accuracy for safety-critical applications. Models must balance *computational efficiency* with detection performance across diverse weather and lighting conditions.",
        "model_suggestions": """Based on the provided ArXiv papers and research context, here are the top 3 YOLO-based object detection models for real-time performance:

## 1. YOLOv8n (Nano)

**Architecture Overview**: Ultralytics YOLOv8 nano variant optimized for edge deployment

- **Performance**: 42.3 mAP on COCO with 31.2 FPS on RTX 3060
- **Model Size**: 6.2MB parameters
- **Strengths**: Best speed-accuracy tradeoff for real-time applications
- **Weaknesses**: Reduced accuracy on small objects compared to larger variants
- **Use Case**: Mobile and edge devices requiring real-time inference

## 2. YOLOv9-C (Compact)

**Architecture Overview**: Next-generation YOLO with Programmable Gradient Information (PGI)

- **Performance**: 47.0 mAP on COCO with 26.8 FPS on RTX 3060
- **Model Size**: 25.3MB parameters  
- **Strengths**: Improved gradient flow and information preservation
- **Weaknesses**: Slightly slower than YOLOv8n but better accuracy
- **Use Case**: Applications requiring higher accuracy with acceptable speed reduction

## 3. RT-DETR-R18 (Real-Time Detection Transformer)

**Architecture Overview**: Transformer-based detector with optimized decoder

- **Performance**: 46.5 mAP on COCO with 20.7 FPS on RTX 3060
- **Model Size**: 20.0MB parameters
- **Strengths**: No NMS post-processing required, stable training
- **Weaknesses**: Lower FPS but excellent accuracy consistency  
- **Use Case**: Research applications and scenarios prioritizing accuracy over maximum speed""",
        "methodology": "The analysis employed a **systematic literature review** approach, analyzing papers from 2023-2024 focusing on real-time object detection. Models were evaluated based on *inference speed*, accuracy metrics (mAP), and computational requirements.",
        "papers_analyzed": [
            {"title": "YOLOv8: A new era of YOLO object detection", "authors": "Jocher, G. et al.", "year": "2023"},
            {"title": "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information", "authors": "Wang, C. et al.", "year": "2024"},
            {"title": "RT-DETR: DETRs Beat YOLOs on Real-time Object Detection", "authors": "Zhao, Y. et al.", "year": "2023"}
        ],
        "technical_analysis": "All recommended models support **TensorRT optimization** for additional 20-30% speedup. Key considerations include *input resolution scaling* (640px standard, 416px for speed, 832px for accuracy) and batch processing capabilities for multiple video streams."
    }
    
    # Generate the report
    output_path = create_ml_research_report(sample_results, "demo_ml_research_report.docx")
    print(f"✅ Demo ML research report created: {output_path}")
    
    return output_path

if __name__ == "__main__":
    demo_integration()
    print("🎉 ML research report integration demo completed!")
