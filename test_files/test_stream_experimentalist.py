#!/usr/bin/env python3
"""Streaming test harness for the experimentalist workflow."""

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
    """Run the streaming experiment suggestion workflow and print key updates."""
    load_dotenv()

    try:
        from aero.experimentalist import stream_experiment_suggestions
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the experimentalist module is properly set up.")
        return None

    # Sample experimental results to feed into the workflow
    experimental_results = {
        "model_performance": {
            "accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.89,
            "f1_score": 0.87,
            "validation_loss": 0.234,
            "training_time": "2.5 hours",
            "inference_time": "0.02 seconds per sample"
        },
        "training_details": {
            "epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "Adam",
            "dataset_size": 10000,
            "train_val_split": "80/20",
            "regularization": "L2 (0.01)"
        },
        "dataset_info": {
            "name": "Custom Image Classification Dataset",
            "classes": ["cats", "dogs", "birds"],
            "class_distribution": {"cats": 4000, "dogs": 3500, "birds": 2500},
            "image_resolution": "224x224",
            "data_augmentation": ["rotation", "flip", "zoom", "brightness"]
        },
        "observed_issues": [
            "Model shows signs of overfitting after epoch 40",
            "Performance drops significantly on out-of-distribution samples",
            "Class imbalance between cats and birds affects precision",
            "Inference time could be improved for real-time applications"
        ],
        "current_architecture": {
            "model_type": "Convolutional Neural Network",
            "layers": [
                "Conv2D(32, 3x3, ReLU)",
                "MaxPool2D(2x2)",
                "Conv2D(64, 3x3, ReLU)", 
                "MaxPool2D(2x2)",
                "Conv2D(128, 3x3, ReLU)",
                "GlobalAveragePooling2D",
                "Dense(128, ReLU)",
                "Dropout(0.5)",
                "Dense(3, Softmax)"
            ],
            "total_parameters": "~2.1M"
        }
    }

    prompt = (
        "Based on my current CNN image classification results, suggest follow-up experiments "
        "to improve model generalization and reduce overfitting. I'm particularly interested "
        "in techniques that could help with out-of-distribution performance and class imbalance issues."
    )

    print("🧪 Streaming Experimentalist Test")
    print("=" * 60)
    print(f"📝 Prompt: {prompt}")
    print()
    print("📊 Experimental Results Summary:")
    print(f"   • Accuracy: {experimental_results['model_performance']['accuracy']}")
    print(f"   • Model Type: {experimental_results['current_architecture']['model_type']}")
    print(f"   • Dataset: {experimental_results['dataset_info']['name']}")
    print(f"   • Key Issues: {len(experimental_results['observed_issues'])} identified")
    print()

    final_result: Optional[Dict[str, Any]] = None
    update_count = 0

    try:
        async for update in stream_experiment_suggestions(
            prompt=prompt,
            experimental_results=experimental_results
        ):
            update_count += 1
            
            # Print status updates
            status = update.get("status") or update.get("current_step", "")
            if status:
                print(f"➡️  Update {update_count}: {status}")
            
            # Print research direction analysis
            if "research_direction" in update and update["research_direction"]:
                direction = update["research_direction"]
                if isinstance(direction, dict) and direction.get("analysis_successful"):
                    print("🔍 Research Direction Analysis:")
                    print(f"   Direction identified and validated")
                    print()
            
            # Print experiment search progress
            if "experiment_papers" in update and update["experiment_papers"]:
                papers = update["experiment_papers"] 
                if isinstance(papers, dict) and papers.get("search_successful"):
                    paper_count = papers.get("papers_returned", 0)
                    print(f"📚 Found {paper_count} relevant research papers")
                    print()
            
            # Print methodology distillation
            if "distilled_methodologies" in update and update["distilled_methodologies"]:
                methodologies = update["distilled_methodologies"]
                if isinstance(methodologies, dict) and methodologies.get("distillation_successful"):
                    method_count = len(methodologies.get("methodologies", []))
                    print(f"⚗️  Distilled {method_count} experimental methodologies")
                    print()
            
            # Print experiment suggestions preview
            if "experiment_suggestions" in update and update["experiment_suggestions"]:
                suggestions = update["experiment_suggestions"]
                if isinstance(suggestions, str) and len(suggestions) > 100:
                    preview = suggestions[:200] + ("..." if len(suggestions) > 200 else "")
                    print("🔬 Experiment Suggestions Preview:")
                    print(preview)
                    print()
                elif isinstance(suggestions, dict):
                    print("🔬 Experiment Suggestions Generated:")
                    if "prioritized_experiments" in suggestions:
                        exp_count = len(suggestions["prioritized_experiments"])
                        print(f"   • {exp_count} prioritized experiments")
                    if "implementation_roadmap" in suggestions:
                        print(f"   • Implementation roadmap included")
                    print()
            
            # Print final summary
            if "experiment_summary" in update and update["experiment_summary"]:
                summary = update["experiment_summary"]
                if isinstance(summary, dict) and summary.get("generation_successful"):
                    print("✅ Experiment Generation Summary:")
                    print(f"   • Status: {summary.get('status', 'completed')}")
                    if "experiments_generated" in summary:
                        print(f"   • Experiments: {summary['experiments_generated']}")
                    if "methodologies_used" in summary:
                        print(f"   • Methodologies: {summary['methodologies_used']}")
                    print()

            final_result = update

    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("=" * 60)
    print("🎯 Final Results:")
    
    if final_result:
        # Print final experiment suggestions
        if "experiment_suggestions" in final_result:
            suggestions = final_result["experiment_suggestions"]
            if isinstance(suggestions, str):
                print("📋 Experiment Suggestions:")
                print(suggestions[:500] + ("..." if len(suggestions) > 500 else ""))
                print()
            elif isinstance(suggestions, dict) and "prioritized_experiments" in suggestions:
                experiments = suggestions["prioritized_experiments"]
                print(f"📋 Generated {len(experiments)} Prioritized Experiments:")
                for i, exp in enumerate(experiments[:3], 1):  # Show first 3
                    title = exp.get("title", f"Experiment {i}")
                    priority = exp.get("priority", "medium")
                    print(f"   {i}. {title} (Priority: {priority})")
                print()
        
        # Print implementation roadmap
        if "implementation_roadmap" in final_result:
            roadmap = final_result["implementation_roadmap"]
            if isinstance(roadmap, dict):
                print("🗺️  Implementation Roadmap Available:")
                if "phases" in roadmap:
                    print(f"   • {len(roadmap['phases'])} implementation phases")
                if "timeline" in roadmap:
                    print(f"   • Estimated timeline: {roadmap['timeline']}")
                print()
        
        # Print processing statistics
        error_count = len(final_result.get("errors", []))
        if error_count > 0:
            print(f"⚠️  {error_count} errors encountered during processing")
        else:
            print("✅ Processing completed without errors")
        
        print(f"📊 Total streaming updates: {update_count}")
        print()
    else:
        print("❌ No final result received")

    return final_result


if __name__ == "__main__":
    """Run the test when script is executed directly."""
    print("Starting Experimentalist Workflow Test...")
    print()
    
    try:
        result = asyncio.run(main())
        if result:
            print("✅ Test completed successfully!")
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