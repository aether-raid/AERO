#!/usr/bin/env python3
"""
Test script for the Experiment Suggestion Workflow

This script tests the experiment suggestion functionality of the AERO framework.
It validates that the workflow can analyze experimental results and suggest
follow-up experiments based on research findings.
"""

import asyncio
import sys
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

async def test_experiment_suggestion():
    """Test the experiment suggestion workflow with sample data."""
    print("🧪 Testing Experiment Suggestion Workflow")
    print("=" * 50)

    try:
        # Import the workflow function
        from src.aero.experimentalist.experiment_suggestion_nodes import run_experiment_suggestion_workflow

        # Sample experimental results for testing
        sample_experimental_results = {
            "model_performance": {
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.89,
                "f1_score": 0.87,
                "validation_loss": 0.234,
                "training_time": "2.5 hours"
            },
            "training_details": {
                "epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 32,
                "optimizer": "Adam",
                "dataset_size": 10000,
                "train_val_split": "80/20"
            },
            "dataset_info": {
                "name": "Custom Image Dataset",
                "classes": ["class_a", "class_b", "class_c"],
                "class_distribution": {"class_a": 4000, "class_b": 3500, "class_c": 2500},
                "image_size": "224x224",
                "preprocessing": ["normalization", "data_augmentation"]
            },
            "observations": [
                "Model converged after 35 epochs",
                "Validation accuracy plateaued around epoch 40",
                "Some overfitting observed in final epochs",
                "Class C has lower performance compared to others",
                "Training was stable with no gradient explosions"
            ],
            "challenges": [
                "Class imbalance in dataset (class C underrepresented)",
                "Limited computational resources for hyperparameter tuning",
                "Dataset size constraints for deep learning",
                "Need for better generalization to unseen data"
            ],
            "current_approach": {
                "model_architecture": "ResNet50 with custom classifier head",
                "framework": "PyTorch",
                "pretrained": True,
                "data_augmentation": ["random_crop", "horizontal_flip", "color_jitter"]
            }
        }

        # Test prompt
        test_prompt = "I have completed initial experiments on image classification with CNNs. My model achieves 87% accuracy but shows signs of overfitting and struggles with class imbalance. Need suggestions for follow-up experiments to improve model performance and generalization."

        print("📝 Test Prompt:")
        print(f"   {test_prompt}")
        print("\n🔬 Sample Experimental Results:")
        print(f"   - Accuracy: {sample_experimental_results['model_performance']['accuracy']}")
        print(f"   - Dataset Size: {sample_experimental_results['training_details']['dataset_size']}")
        print(f"   - Classes: {len(sample_experimental_results['dataset_info']['classes'])}")
        print(f"   - Challenges: {len(sample_experimental_results['challenges'])} identified")

        print("\n🚀 Starting Experiment Suggestion Workflow...")

        # Run the workflow
        result = await run_experiment_suggestion_workflow(
            user_prompt=test_prompt,
            experimental_results=sample_experimental_results
        )

        print("\n✅ Workflow completed successfully!")

        # Validate results
        if not result:
            print("❌ No result returned from workflow")
            return None

        # Check for errors
        errors = result.get("errors", [])
        if errors:
            print(f"⚠️ Workflow completed with {len(errors)} errors:")
            for i, error in enumerate(errors[-3:], 1):  # Show last 3 errors
                print(f"  {i}. {error}")

        # Show workflow statistics
        print("\n📊 WORKFLOW STATISTICS:")
        print(f"   - Analysis iterations: {len(result.get('analysis_iterations', []))}")
        print(f"   - Direction iterations: {len(result.get('direction_iterations', []))}")
        print(f"   - Search iterations: {result.get('experiment_search_iteration', 0)}")
        print(f"   - Experiment iterations: {result.get('current_experiment_iteration', 0)}")
        print(f"   - Papers analyzed: {len(result.get('validated_experiment_papers', []))}")

        # Check for experiment suggestions
        experiment_suggestions = result.get("experiment_suggestions", "")
        if experiment_suggestions:
            print("\n✅ Experiment suggestions generated successfully!")
            print(f"   - Length: {len(experiment_suggestions)} characters")

            # Show preview of suggestions
            preview = experiment_suggestions[:500] + "..." if len(experiment_suggestions) > 500 else experiment_suggestions
            print(f"   - Preview: {preview.replace(chr(10), ' ').replace(chr(13), ' ')}")

            # Check for key components in suggestions
            key_indicators = [
                "experiment" in experiment_suggestions.lower(),
                "suggestion" in experiment_suggestions.lower(),
                "methodology" in experiment_suggestions.lower() or "approach" in experiment_suggestions.lower()
            ]

            if any(key_indicators):
                print("   ✅ Suggestions contain relevant experimental content")
            else:
                print("   ⚠️ Suggestions may lack specific experimental details")

        else:
            print("❌ No experiment suggestions generated")
            return None

        # Check for prioritized experiments
        prioritized_experiments = result.get("prioritized_experiments", [])
        if prioritized_experiments:
            print(f"\n📋 Prioritized Experiments: {len(prioritized_experiments)} suggestions")
            for i, exp in enumerate(prioritized_experiments[:3], 1):  # Show first 3
                title = exp.get("title", "Untitled")[:60]
                priority = exp.get("priority", "Unknown")
                print(f"   {i}. [{priority}] {title}")

        # Check for implementation roadmap
        roadmap = result.get("implementation_roadmap", {})
        if roadmap:
            print(f"\n🗺️ Implementation Roadmap: Available")
            phases = roadmap.get("phases", [])
            if phases:
                print(f"   - Phases: {len(phases)}")
                for phase in phases[:2]:  # Show first 2 phases
                    print(f"     • {phase.get('name', 'Unnamed Phase')}")

        # Check for research direction
        research_direction = result.get("research_direction", {})
        if research_direction:
            selected_direction = research_direction.get("selected_direction", {})
            direction_name = selected_direction.get("direction", "Not specified")
            print(f"\n🎯 Research Direction: {direction_name}")

        # Check for findings analysis
        findings_analysis = result.get("findings_analysis", {})
        if findings_analysis:
            domain_analysis = findings_analysis.get("domain_analysis", {})
            primary_domain = domain_analysis.get("primary_domain", "Unknown")
            task_type = domain_analysis.get("task_type", "Unknown")
            print(f"\n🔍 Analysis Results:")
            print(f"   - Primary Domain: {primary_domain}")
            print(f"   - Task Type: {task_type}")

        print("\n🎉 Test completed successfully!")
        return result

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're running this from the project root directory")
        print("and that all dependencies are installed.")
        return None

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function."""
    print("🧪 AERO Experiment Suggestion Workflow Test")
    print("=" * 60)

    result = await test_experiment_suggestion()

    if result:
        print("\n✅ All tests passed! Experiment suggestion workflow is working correctly.")
    else:
        print("\n❌ Test failed! Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())