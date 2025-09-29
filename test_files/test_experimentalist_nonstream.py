#!/usr/bin/env python3
"""Non-streaming test harness for the experimentalist workflow."""

import asyncio
import os
import sys
from typing import Optional, Dict, Any

from dotenv import load_dotenv


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.join(ROOT_DIR, "..", "src")
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)


async def main() -> Optional[Dict[str, Any]]:
    """Run the non-streaming experiment suggestion workflow."""
    load_dotenv()

    try:
        from aero.experimentalist import suggest_experiments_nostream
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the experimentalist module is properly set up.")
        return None

    # Sample experimental results focused on a specific research problem
    experimental_results = {
        "research_context": {
            "domain": "Computer Vision",
            "task": "Object Detection in Autonomous Vehicles",
            "current_approach": "YOLO-based detection pipeline",
            "performance_target": "Real-time inference (<50ms) with >95% accuracy"
        },
        "current_results": {
            "accuracy": 0.91,
            "inference_time": "65ms per frame",
            "false_positive_rate": 0.08,
            "false_negative_rate": 0.05,
            "model_size": "180MB"
        },
        "experimental_setup": {
            "dataset": "Custom automotive dataset (50K images)",
            "hardware": "NVIDIA RTX 3080",
            "framework": "PyTorch",
            "training_time": "18 hours",
            "validation_split": "85/15"
        },
        "identified_challenges": [
            "Inference time exceeds real-time requirements",
            "Performance degrades in low-light conditions",
            "Struggles with small object detection (pedestrians at distance)",
            "Model size too large for edge deployment",
            "High false positive rate for similar-looking objects"
        ],
        "research_questions": [
            "How can we reduce inference time while maintaining accuracy?",
            "What architectural changes could improve small object detection?",
            "Which data augmentation techniques work best for automotive scenarios?",
            "How can we make the model more robust to lighting variations?",
            "What quantization techniques preserve accuracy for edge deployment?"
        ],
        "constraints": {
            "hardware_budget": "Limited to consumer-grade GPUs",
            "deployment_target": "Embedded automotive systems",
            "timeline": "6-month development cycle",
            "data_availability": "Cannot collect significantly more labeled data"
        }
    }

    prompt = (
        "Design a comprehensive experimental plan to optimize our YOLO-based object detection "
        "system for real-time autonomous vehicle applications. Focus on reducing inference time "
        "to under 50ms while maintaining >95% accuracy, with particular attention to small object "
        "detection and robustness to lighting conditions."
    )

    print("🧪 Non-Streaming Experimentalist Test")
    print("=" * 60)
    print(f"📝 Research Prompt: {prompt}")
    print()
    print("📊 Current System Performance:")
    print(f"   • Accuracy: {experimental_results['current_results']['accuracy']}")
    print(f"   • Inference Time: {experimental_results['current_results']['inference_time']}")
    print(f"   • Model Size: {experimental_results['current_results']['model_size']}")
    print(f"   • Target: <50ms inference, >95% accuracy")
    print()
    print("🎯 Key Research Questions:")
    for i, question in enumerate(experimental_results['research_questions'][:3], 1):
        print(f"   {i}. {question}")
    print(f"   ... and {len(experimental_results['research_questions'])-3} more")
    print()
    
    print("🔄 Running experiment suggestion workflow...")
    print("   This may take a few minutes as it searches literature and analyzes methodologies...")
    print()

    try:
        result = await suggest_experiments_nostream(
            prompt=prompt,
            experimental_results=experimental_results
        )
        
        print("✅ Workflow completed successfully!")
        print("=" * 60)
        
        if result and "experiment_suggestions" in result:
            suggestions = result["experiment_suggestions"]
            
            print("📋 Generated Experiment Suggestions:")
            print("-" * 40)
            if isinstance(suggestions, str):
                # Print first part of suggestions
                lines = suggestions.split('\n')
                for line in lines[:20]:  # First 20 lines
                    if line.strip():
                        print(line)
                if len(lines) > 20:
                    print(f"\n... ({len(lines)-20} more lines)")
            elif isinstance(suggestions, dict):
                if "prioritized_experiments" in suggestions:
                    experiments = suggestions["prioritized_experiments"]
                    print(f"🔬 {len(experiments)} Prioritized Experiments:")
                    for i, exp in enumerate(experiments, 1):
                        title = exp.get("title", f"Experiment {i}")
                        priority = exp.get("priority", "medium")
                        rationale = exp.get("rationale", "No rationale provided")
                        print(f"\n{i}. {title} (Priority: {priority})")
                        print(f"   Rationale: {rationale[:100]}...")
            print()
            
        # Print implementation roadmap if available
        if result and "implementation_roadmap" in result:
            roadmap = result["implementation_roadmap"]
            if isinstance(roadmap, dict):
                print("🗺️  Implementation Roadmap:")
                print("-" * 40)
                if "phases" in roadmap:
                    phases = roadmap["phases"]
                    for i, phase in enumerate(phases, 1):
                        phase_name = phase.get("name", f"Phase {i}")
                        duration = phase.get("duration", "TBD")
                        print(f"{i}. {phase_name} ({duration})")
                        if "tasks" in phase:
                            tasks = phase["tasks"]
                            for task in tasks[:2]:  # Show first 2 tasks
                                print(f"   • {task}")
                            if len(tasks) > 2:
                                print(f"   • ... and {len(tasks)-2} more tasks")
                print()
        
        # Print analysis summary
        if result and "experiment_summary" in result:
            summary = result["experiment_summary"]
            if isinstance(summary, dict):
                print("📈 Analysis Summary:")
                print("-" * 40)
                if "papers_analyzed" in summary:
                    print(f"Research Papers Analyzed: {summary['papers_analyzed']}")
                if "methodologies_identified" in summary:
                    print(f"Methodologies Identified: {summary['methodologies_identified']}")
                if "experiments_proposed" in summary:
                    print(f"Experiments Proposed: {summary['experiments_proposed']}")
                if "estimated_timeline" in summary:
                    print(f"Estimated Timeline: {summary['estimated_timeline']}")
                print()
        
        # Print any errors encountered
        if result and "errors" in result and result["errors"]:
            print("⚠️  Errors Encountered:")
            print("-" * 40)
            for error in result["errors"][:3]:  # Show first 3 errors
                print(f"• {error}")
            if len(result["errors"]) > 3:
                print(f"• ... and {len(result['errors'])-3} more errors")
            print()
        
        return result
        
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """Run the test when script is executed directly."""
    print("Starting Non-Streaming Experimentalist Test...")
    print()
    
    try:
        result = asyncio.run(main())
        if result:
            print("✅ Test completed successfully!")
            print("\n💡 Tip: Check the generated experiment suggestions for actionable research directions.")
        else:
            print("❌ Test failed or returned no results")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)