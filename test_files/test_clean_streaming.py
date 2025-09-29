#!/usr/bin/env python3
"""
Simple Research Planning Library Test - FIXED VERSION
====================================================

Clean, simple test using only the plan_research function with streaming support.
"""

from dotenv import load_dotenv
import os
import sys
import asyncio

async def main():
    """Simple test of research planning library with streaming."""
    
    print("🔧 Loading environment variables...")
    
    # Load .env file into environment variables
    load_dotenv()
    
    # Verify key variables are loaded
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("DEFAULT_MODEL")
    
    print(f"✅ OPENAI_API_KEY: {'Loaded' if api_key else 'Missing'}")
    print(f"✅ BASE_URL: {base_url or 'Not set'}")
    print(f"✅ DEFAULT_MODEL: {model or 'Not set'}")
    
    if not api_key:
        print("❌ OPENAI_API_KEY is required! Please check your .env file.")
        return
    
    print("\n📦 Importing research planning library...")
    
    try:
        # Import the plan_research function
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
        from research_planner.main import plan_research
        
        print("✅ Library imported successfully!")
        print("\n🚀 Running research planning test with streaming...")
        
        # Simple test query
        query = "What are the problems in using government data for machine learning?"
        print(f"📝 Query: {query}")
        
        # Call plan_research with streaming enabled
        result = await plan_research(query, enable_streaming=True)
        
        # Check results
        if result and result.get("research_plan"):
            print("\n✅ Research planning completed successfully!")
            plan_data = result["research_plan"]
            
            # Extract the actual research plan content
            if isinstance(plan_data, dict) and "research_plan" in plan_data:
                research_plan_content = plan_data.get("research_plan", "")
                
                if research_plan_content:
                    # Try to extract title from the content
                    lines = research_plan_content.split('\n')
                    title = "Research Plan"
                    for line in lines[:15]:  # Check first 15 lines for title
                        if line.strip().startswith('#') and 'summary' not in line.lower():
                            title = line.strip('#').strip()
                            break
                    
                    # Count sections by looking for headers
                    section_count = len([line for line in lines if line.strip().startswith('##')])
                    
                    print(f"📋 Title: {title}")
                    print(f"📊 Sections: {section_count} parts")
                    print(f"📄 Content length: {len(research_plan_content):,} characters")
                    print(f"✅ Generation status: {plan_data.get('research_plan_successful', 'Unknown')}")
                    
                    # Show first few lines as preview
                    preview_lines = research_plan_content.split('\n')[:5]
                    print(f"\n📝 Research Plan Preview:")
                    for line in preview_lines:
                        if line.strip():
                            print(f"   {line}")
                    print("   ...")
                    
                else:
                    print(f"📋 Title: N/A")
                    print(f"📊 Sections: 0 parts") 
                    print("⚠️ Research plan content is empty")
            else:
                # Fallback for other formats
                title = plan_data.get('title', 'N/A')
                sections = plan_data.get('sections', [])
                print(f"� Title: {title}")
                print(f"�📊 Sections: {len(sections)} parts")
            
            if result.get("display_method") == "terminal_output":
                print("📄 Research plan was displayed above with streaming")
        elif result and result.get("error"):
            print(f"\n❌ Research planning failed with error: {result.get('error')}")
        else:
            print("\n❌ Research planning failed or returned no results")
            print(f"   Available keys: {list(result.keys()) if result else 'None'}")
            if result:
                print(f"   Result type: {type(result)}")
                if result.get("research_plan"):
                    plan_data = result["research_plan"]
                    print(f"   Plan data type: {type(plan_data)}")
                    if isinstance(plan_data, dict):
                        print(f"   Plan data keys: {list(plan_data.keys())}")
            
    except ImportError as e:
        print(f"❌ Failed to import library: {e}")
        print("   Make sure you're in the correct directory")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())