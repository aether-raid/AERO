#!/usr/bin/env python3
"""
Paper Writing Workflow Test with Streaming
==========================================

Clean test file for the paper writing workflow with real-time streaming support.
"""

from dotenv import load_dotenv
import os
import sys
import asyncio

async def main():
    """Test the paper writing workflow with streaming."""
    
    print("📝 Paper Writing Workflow Test")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Verify key variables are loaded
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    print(f"✅ OPENAI_API_KEY: {'Loaded' if api_key else 'Missing'}")
    print(f"✅ BASE_URL: {base_url or 'Not set'}")
    print(f"✅ TAVILY_API_KEY: {'Loaded' if tavily_key else 'Missing'}")
    
    if not api_key:
        print("❌ OPENAI_API_KEY is required! Please check your .env file.")
        return
    
    print("\n📦 Importing paper writing library...")
    
    try:
        # Import the paper writing function
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'aero', 'report_writer'))
        from paper_writing_nodes import write_paper
        
        print("✅ Library imported successfully!")
        print("\n🚀 Running paper writing test with streaming...")
        
        # Test query with experimental data
        query = "Write a research paper analyzing machine learning model performance"
        experimental_data = {
            "model_type": "Random Forest",
            "accuracy": 0.95,
            "f1_score": 0.92,
            "precision": 0.94,
            "recall": 0.90,
            "dataset_size": 10000,
            "training_time": "2.5 hours",
            "cross_validation_scores": [0.94, 0.96, 0.93, 0.95, 0.92]
        }
        
        print(f"📝 Query: {query}")
        print(f"📊 Experimental data: {len(experimental_data)} metrics")
        
        # Call write_paper with streaming enabled
        result = await write_paper(
            user_query=query, 
            experimental_data=experimental_data,
            target_venue="ACM Conference",
            enable_streaming=True
        )
        
        # Check results
        if result and result.get("formatted_paper"):
            print("\n✅ Paper writing completed successfully!")
            
            # Extract paper information
            paper_content = result.get("formatted_paper", "")
            
            if paper_content:
                # Try to extract title from content
                lines = paper_content.split('\n')
                title = "Research Paper"
                for line in lines[:10]:
                    if line.strip() and not line.startswith('#') and len(line.strip()) > 5:
                        title = line.strip()
                        break
                
                # Count sections by looking for headers
                section_count = len([line for line in lines if line.strip().startswith('##')])
                
                print(f"📋 Title: {title[:80]}{'...' if len(title) > 80 else ''}")
                print(f"📊 Sections: {section_count} parts")
                print(f"📄 Content length: {len(paper_content):,} characters")
                
                # Show paper preview
                print(f"\n📝 Paper Preview:")
                preview_lines = [line for line in lines[:8] if line.strip()][:5]
                for line in preview_lines:
                    if line.strip():
                        print(f"   {line.strip()}")
                print("   ...")
                
                # Show additional details
                if result.get("supporting_sources"):
                    sources_count = len(result["supporting_sources"])
                    print(f"📚 References: {sources_count} sources found")
                
                if result.get("critique_results"):
                    critique = result["critique_results"]
                    if isinstance(critique, dict) and critique.get("quality_score"):
                        print(f"📊 Quality score: {critique['quality_score']}/10")
                
            else:
                print("⚠️ Paper content is empty")
                
        elif result and result.get("error"):
            print(f"\n❌ Paper writing failed with error: {result.get('error')}")
        else:
            print("\n❌ Paper writing failed or returned no results")
            if result:
                print(f"   Available keys: {list(result.keys())}")
                print(f"   Current step: {result.get('current_step', 'Unknown')}")
                if result.get("errors"):
                    print(f"   Errors: {result['errors']}")
            
    except ImportError as e:
        print(f"❌ Failed to import library: {e}")
        print("   Make sure you're in the correct directory and dependencies are installed")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())