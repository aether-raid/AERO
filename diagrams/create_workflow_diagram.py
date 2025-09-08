#!/usr/bin/env python3
"""
Complete Workflow Diagram for ML Researcher LangGraph Multi-Workflow System
This creates a visual representation of the entire system architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_workflow_diagram():
    """Create a comprehensive workflow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'router': '#FF6B6B',      # Red for router
        'model': '#4ECDC4',       # Teal for model suggestion
        'research': '#45B7D1',    # Blue for research planning
        'validation': '#96CEB4',  # Green for validation
        'decision': '#FECA57',    # Yellow for decisions
        'output': '#DDA0DD',      # Purple for outputs
        'input': '#FFE4B5'        # Light orange for inputs
    }
    
    # Title
    ax.text(10, 15.5, 'ML Researcher LangGraph - Complete Workflow Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # === USER INPUT SECTION ===
    user_input = FancyBboxPatch((0.5, 14), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(user_input)
    ax.text(2, 14.5, 'User Input\n(Research Query)', fontsize=10, ha='center', va='center', fontweight='bold')
    
    # === ROUTER SECTION ===
    router_box = FancyBboxPatch((8, 12.5), 4, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['router'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(router_box)
    ax.text(10, 13.5, 'ROUTER AGENT\n\nAnalyzes query type\nRoutes to appropriate workflow\nConfidence scoring', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # === MODEL SUGGESTION WORKFLOW ===
    # Header
    ax.text(4, 11, 'MODEL SUGGESTION WORKFLOW', fontsize=14, fontweight='bold', ha='center', color=colors['model'])
    
    # Model workflow nodes
    model_nodes = [
        (1, 9.5, 'Search arXiv\nPapers'),
        (1, 8, 'Process\nPDFs'),
        (1, 6.5, 'Analyze\nDomain'),
        (1, 5, 'Generate Model\nSuggestions'),
        (1, 3.5, 'Final\nRecommendations')
    ]
    
    for x, y, text in model_nodes:
        node = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor=colors['model'], 
                             edgecolor='black')
        ax.add_patch(node)
        ax.text(x, y, text, fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Model workflow arrows
    for i in range(len(model_nodes)-1):
        ax.arrow(1, model_nodes[i][1]-0.4, 0, -0.7, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # === RESEARCH PLANNING WORKFLOW (ITERATIVE) ===
    # Header
    ax.text(16, 11, 'RESEARCH PLANNING WORKFLOW\n(Iterative Problem Validation)', 
            fontsize=14, fontweight='bold', ha='center', color=colors['research'])
    
    # Main iterative loop
    loop_center_x, loop_center_y = 16, 8
    loop_radius = 2.5
    
    # Generate Problem Node
    gen_node = FancyBboxPatch((loop_center_x-3, loop_center_y+1.5), 2, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor=colors['research'], 
                             edgecolor='black')
    ax.add_patch(gen_node)
    ax.text(loop_center_x-2, loop_center_y+1.9, 'Generate\nProblem', fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Validate Problem Node
    val_node = FancyBboxPatch((loop_center_x+1, loop_center_y+1.5), 2, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor=colors['validation'], 
                             edgecolor='black')
    ax.add_patch(val_node)
    ax.text(loop_center_x+2, loop_center_y+1.9, 'Validate Problem\n(LLM Knowledge)', fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Decision Diamond - Accept/Reject
    decision1 = patches.RegularPolygon((loop_center_x+2, loop_center_y), 4, radius=0.5, 
                                      orientation=np.pi/4, facecolor=colors['decision'], 
                                      edgecolor='black')
    ax.add_patch(decision1)
    ax.text(loop_center_x+2, loop_center_y, 'Accept?', fontsize=7, ha='center', va='center', fontweight='bold')
    
    # Collect Problem Node
    collect_node = FancyBboxPatch((loop_center_x-3, loop_center_y-1.5), 2, 0.8, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors['research'], 
                                 edgecolor='black')
    ax.add_patch(collect_node)
    ax.text(loop_center_x-2, loop_center_y-1.1, 'Collect\nProblem', fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Decision Diamond - Continue/Finish
    decision2 = patches.RegularPolygon((loop_center_x-2, loop_center_y-3), 4, radius=0.5, 
                                      orientation=np.pi/4, facecolor=colors['decision'], 
                                      edgecolor='black')
    ax.add_patch(decision2)
    ax.text(loop_center_x-2, loop_center_y-3, 'Need\nMore?', fontsize=7, ha='center', va='center', fontweight='bold')
    
    # Final Research Plan Node
    final_node = FancyBboxPatch((loop_center_x+1, loop_center_y-3), 2, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['output'], 
                               edgecolor='black')
    ax.add_patch(final_node)
    ax.text(loop_center_x+2, loop_center_y-2.6, 'Create Research\nPlan', fontsize=8, ha='center', va='center', fontweight='bold')
    
    # === ROUTING ARROWS ===
    # User to Router
    ax.arrow(3.5, 14.5, 4, -1, head_width=0.15, head_length=0.2, fc='black', ec='black', linewidth=2)
    
    # Router to Model Workflow
    ax.arrow(8.5, 12.8, -6.5, -2.5, head_width=0.15, head_length=0.2, fc=colors['model'], ec=colors['model'], linewidth=2)
    ax.text(5, 11.5, 'Model/Architecture\nQueries', fontsize=8, ha='center', color=colors['model'], fontweight='bold')
    
    # Router to Research Workflow
    ax.arrow(11.5, 12.8, 2.5, -2.5, head_width=0.15, head_length=0.2, fc=colors['research'], ec=colors['research'], linewidth=2)
    ax.text(13, 11.5, 'Research Problem\nQueries', fontsize=8, ha='center', color=colors['research'], fontweight='bold')
    
    # === ITERATIVE LOOP ARROWS ===
    # Generate → Validate
    ax.arrow(loop_center_x-1, loop_center_y+1.9, 1.5, 0, head_width=0.1, head_length=0.15, fc='blue', ec='blue')
    
    # Validate → Decision1
    ax.arrow(loop_center_x+2, loop_center_y+1.1, 0, 0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    # Decision1 → Collect (Accept)
    ax.arrow(loop_center_x+1.5, loop_center_y-0.3, -2, -0.8, head_width=0.1, head_length=0.1, fc='green', ec='green')
    ax.text(loop_center_x-0.5, loop_center_y-0.8, 'Accept', fontsize=7, color='green', fontweight='bold')
    
    # Decision1 → Generate (Reject - loop back)
    ax.arrow(loop_center_x+1.5, loop_center_y+0.3, -4, 1.2, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax.text(loop_center_x-1, loop_center_y+0.8, 'Reject\n(Loop)', fontsize=7, color='red', fontweight='bold')
    
    # Collect → Decision2
    ax.arrow(loop_center_x-2, loop_center_y-1.9, 0, -0.6, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    # Decision2 → Generate (Need More)
    ax.arrow(loop_center_x-2.5, loop_center_y-2.7, 0, 3.7, head_width=0.1, head_length=0.1, fc='orange', ec='orange')
    ax.text(loop_center_x-3.5, loop_center_y, 'Need More\n(< 3 problems)', fontsize=7, color='orange', fontweight='bold', rotation=90)
    
    # Decision2 → Final (Enough)
    ax.arrow(loop_center_x-1.5, loop_center_y-3, 1, 0, head_width=0.1, head_length=0.15, fc='purple', ec='purple')
    ax.text(loop_center_x-0.5, loop_center_y-3.5, 'Enough\n(≥3 problems)', fontsize=7, color='purple', fontweight='bold')
    
    # === OUTPUT SECTION ===
    # Model Output
    model_output = FancyBboxPatch((0.5, 1.5), 3, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['output'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(model_output)
    ax.text(2, 2.25, 'MODEL OUTPUT:\n• Architecture recommendations\n• Performance comparisons\n• Implementation guidance', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Research Output
    research_output = FancyBboxPatch((16.5, 1.5), 3, 1.5, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['output'], 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(research_output)
    ax.text(18, 2.25, 'RESEARCH OUTPUT:\n• Validated open problems\n• Comprehensive research plan\n• Timeline & resources', 
            fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Output arrows
    ax.arrow(1, 3.1, 0, -1.2, head_width=0.15, head_length=0.2, fc=colors['output'], ec=colors['output'], linewidth=2)
    ax.arrow(loop_center_x+2, loop_center_y-3.4, 0.5, -1.5, head_width=0.15, head_length=0.2, fc=colors['output'], ec=colors['output'], linewidth=2)
    
    # === LEGEND ===
    legend_y = 0.5
    legend_items = [
        (colors['router'], 'Router & Decision Logic'),
        (colors['model'], 'Model Suggestion Workflow'),
        (colors['research'], 'Research Planning Workflow'),
        (colors['validation'], 'Problem Validation'),
        (colors['decision'], 'Conditional Decisions'),
        (colors['output'], 'Final Outputs')
    ]
    
    ax.text(10, legend_y + 0.5, 'LEGEND:', fontsize=12, fontweight='bold', ha='center')
    for i, (color, label) in enumerate(legend_items):
        x_pos = 7 + (i % 3) * 2.5
        y_pos = legend_y - 0.3 if i < 3 else legend_y - 0.6
        
        legend_box = FancyBboxPatch((x_pos-0.1, y_pos-0.1), 0.2, 0.2, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor=color, 
                                   edgecolor='black')
        ax.add_patch(legend_box)
        ax.text(x_pos+0.3, y_pos, label, fontsize=9, va='center')
    
    # === KEY FEATURES ANNOTATIONS ===
    # Iterative Loop Annotation
    ax.annotate('ITERATIVE VALIDATION LOOP:\n• Generate problems\n• Validate against existing solutions\n• Collect only novel problems\n• Continue until 3+ problems found', 
                xy=(loop_center_x, loop_center_y), xytext=(12, 6),
                fontsize=9, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='orange'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='orange', lw=2))
    
    # Router Intelligence Annotation
    ax.annotate('INTELLIGENT ROUTING:\n• Analyzes query semantics\n• Confidence scoring\n• Automatic workflow selection', 
                xy=(10, 13.5), xytext=(6, 13),
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', edgecolor='red'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', color='red', lw=2))
    
    plt.tight_layout()
    return fig

def create_state_flow_diagram():
    """Create a detailed state flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(9, 11.5, 'State Flow & Data Structures', fontsize=18, fontweight='bold', ha='center')
    
    # === ROUTER STATE ===
    ax.text(3, 10.5, 'RouterState', fontsize=14, fontweight='bold', ha='center', color='red')
    router_state = """
    {
        "messages": [],
        "original_prompt": str,
        "workflow_decision": str,
        "routing_confidence": float,
        "routing_reasoning": str,
        "errors": []
    }
    """
    ax.text(3, 9.5, router_state, fontsize=8, ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    # === MODEL SUGGESTION STATE ===
    ax.text(9, 10.5, 'ModelSuggestionState', fontsize=14, fontweight='bold', ha='center', color='teal')
    model_state = """
    {
        "messages": [],
        "original_prompt": str,
        "arxiv_results": dict,
        "domain_analysis": dict,
        "model_suggestions": dict,
        "current_step": str,
        "errors": []
    }
    """
    ax.text(9, 9.5, model_state, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    # === RESEARCH PLANNING STATE ===
    ax.text(15, 10.5, 'ResearchPlanningState', fontsize=14, fontweight='bold', ha='center', color='blue')
    research_state = """
    {
        "messages": [],
        "original_prompt": str,
        "generated_problems": [],
        "validated_problems": [],
        "current_problem": dict,
        "validation_results": dict,
        "iteration_count": int,
        "research_plan": dict,
        "current_step": str,
        "errors": []
    }
    """
    ax.text(15, 9.5, research_state, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightsteelblue'))
    
    # === VALIDATION STRUCTURE ===
    ax.text(9, 7, 'Problem Validation Structure', fontsize=14, fontweight='bold', ha='center', color='green')
    validation_structure = """
    Problem = {
        "statement": str,
        "description": str,
        "research_question": str,
        "keywords": [str],
        "validation": {
            "status": "solved|well_studied|partially_solved|open",
            "confidence": float,
            "reasoning": str,
            "existing_solutions": [str],
            "research_gaps": [str],
            "recommendation": "accept|reject"
        }
    }
    """
    ax.text(9, 5.5, validation_structure, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # === CONDITIONAL LOGIC ===
    ax.text(4.5, 3, 'Conditional Logic', fontsize=14, fontweight='bold', ha='center', color='orange')
    conditions = """
    _should_continue_generating():
    • If validated_problems >= 3: "finalize_plan"
    • If iteration_count >= 10: "finalize_plan" 
    • Else: "generate_problem"
    
    _check_completion():
    • If recommendation == "accept": "collect_problem"
    • Else: "continue_generation"
    """
    ax.text(4.5, 1.8, conditions, fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # === OUTPUT FORMAT ===
    ax.text(13.5, 3, 'Final Output Structure', fontsize=14, fontweight='bold', ha='center', color='purple')
    output_format = """
    Results = {
        "workflow_type": str,
        "router_decision": dict,
        "original_prompt": str,
        "validated_problems": [Problem],
        "research_plan": dict,
        "iteration_count": int,
        "summary": {
            "problems_generated": int,
            "problems_validated": int,
            "iterations_completed": int
        }
    }
    """
    ax.text(13.5, 1.8, output_format, fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='plum'))
    
    plt.tight_layout()
    return fig

def save_diagrams():
    """Save both diagrams."""
    print("🎨 Creating ML Researcher LangGraph Workflow Diagrams...")
    
    # Create and save main workflow diagram
    fig1 = create_workflow_diagram()
    fig1.savefig('ml_researcher_workflow_diagram.png', dpi=300, bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
    print("✅ Saved: ml_researcher_workflow_diagram.png")
    
    # Create and save state flow diagram
    fig2 = create_state_flow_diagram()
    fig2.savefig('ml_researcher_state_flow_diagram.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("✅ Saved: ml_researcher_state_flow_diagram.png")
    
    plt.show()
    
    print("\n📊 DIAGRAM SUMMARY:")
    print("=" * 50)
    print("1. ml_researcher_workflow_diagram.png")
    print("   • Complete system architecture")
    print("   • Router logic and workflow routing")
    print("   • Iterative research planning loop")
    print("   • Model suggestion pipeline")
    print("   • Input/output flows")
    print("\n2. ml_researcher_state_flow_diagram.png")
    print("   • Detailed state structures")
    print("   • Data type definitions") 
    print("   • Conditional logic")
    print("   • Problem validation format")
    print("   • Final output structure")

if __name__ == "__main__":
    save_diagrams()
