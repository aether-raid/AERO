#!/usr/bin/env python3
"""Simple test to verify both __init__.py modules work correctly."""

import sys

def test_both_modules():
    print("🧪 Testing Paper Writing Module")
    print("-" * 40)
    
    sys.path.insert(0, './report_writing')
    import __init__ as rw
    
    print(f"✅ Name: {rw.__package_info__['name']}")
    print(f"✅ Version: {rw.__version__}")
    print(f"✅ Features: {len(rw.__package_info__['features'])}")
    print(f"✅ Functions: {len(rw.__all__)}")
    print(f"✅ Key functions: {rw.__all__[:3]}")
    
    print("\n🧪 Testing Research Planner Module")
    print("-" * 40)
    
    sys.path.insert(0, './research_planner')
    import __init__ as rp
    
    print(f"✅ Name: {rp.__package_info__['name']}")
    print(f"✅ Version: {rp.__version__}")
    print(f"✅ Features: {len(rp.__package_info__['features'])}")
    print(f"✅ Functions: {len(rp.__all__)}")
    print(f"✅ Key functions: {rp.__all__[:3]}")
    
    print("\n🎉 Both modules initialized successfully!")
    print(f"📝 Paper Writing: {len(rw.__package_info__['features'])} features, {len(rw.__all__)} functions")
    print(f"🔬 Research Planner: {len(rp.__package_info__['features'])} features, {len(rp.__all__)} functions")

if __name__ == "__main__":
    test_both_modules()
