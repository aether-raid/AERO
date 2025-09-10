#!/usr/bin/env python3
"""
Comprehensive Flow Chart Generator for ML Research Assistant LangGraph Workflows
Creates detailed visual flow charts showing the complete system architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np

def create_comprehensive_flow_chart():
    """Create a comprehensive flow chart of the entire ML Research Assistant system."""
    
    # Create figure with multiple subplots for different workflow views
    fig = plt.figure(figsize=(20, 24))
    
    # Main system overview
    ax1 = plt.subplot(3, 1, 1)
    create_main_system_flow(ax1)
    
    # Model suggestion workflow details
    ax2 = plt.subplot(3, 2, 3)
    create_model_suggestion_flow(ax2)
    
    # Research planning workflow details
    ax3 = plt.subplot(3, 2, 4)
    create_research_planning_flow(ax3)
    
    # Router decision logic
    ax4 = plt.subplot(3, 2, 5)
    create_router_decision_flow(ax4)
    
    # State management flow
    ax5 = plt.subplot(3, 2, 6)
    create_state_management_flow(ax5)
    
    plt.tight_layout()
    plt.savefig('ml_research_workflow_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_main_system_flow(ax):
    """Create the main system overview flow chart."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_title('🌟 ML Research Assistant - Complete System Flow', fontsize=16, fontweight='bold', pad=20)
    
    # Color scheme
    colors = {
        'input': '#E3F2FD',      # Light blue
        'router': '#FFF3E0',     # Light orange
        'model': '#E8F5E8',      # Light green
        'research': '#F3E5F5',   # Light purple
        'decision': '#FFF9C4',   # Light yellow
        'output': '#FFEBEE'      # Light red
    }
    
    # User Input
    user_box = FancyBboxPatch((1, 10.5), 8, 1, boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(user_box)
    ax.text(5, 11, '👤 User Input\nResearch Query/Question', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Router
    router_box = FancyBboxPatch((3.5, 8.5), 3, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['router'], edgecolor='black', linewidth=2)
    ax.add_patch(router_box)
    ax.text(5, 9.25, '🤖 Intelligent Router\nSemantic Analysis\nIntent Classification', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Decision Diamond
    decision_points = np.array([[5, 7.5], [4, 6.5], [5, 5.5], [6, 6.5]])
    decision_diamond = patches.Polygon(decision_points, closed=True, 
                                     facecolor=colors['decision'], edgecolor='black', linewidth=2)
    ax.add_patch(decision_diamond)
    ax.text(5, 6.5, '🎯\nWorkflow\nDecision', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Model Suggestion Workflow
    model_box = FancyBboxPatch((0.5, 3), 3.5, 2, boxstyle="round,pad=0.1",
                               facecolor=colors['model'], edgecolor='black', linewidth=2)
    ax.add_patch(model_box)
    ax.text(2.25, 4, '📊 Model Suggestion\nWorkflow\n\n• Literature Search\n• Domain Analysis\n• Model Recommendations\n• Quality Critique', 
            ha='center', va='center', fontsize=9)
    
    # Research Planning Workflow  
    research_box = FancyBboxPatch((6, 3), 3.5, 2, boxstyle="round,pad=0.1",
                                  facecolor=colors['research'], edgecolor='black', linewidth=2)
    ax.add_patch(research_box)
    ax.text(7.75, 4, '🧪 Research Planning\nWorkflow\n\n• Problem Generation\n• Validation & Selection\n• Research Plan\n• Iterative Refinement', 
            ha='center', va='center', fontsize=9)
    
    # Output
    output_box = FancyBboxPatch((3, 0.5), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 1.25, '📋 Structured Output\nJSON Results with\nDetailed Analysis', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    arrows = [
        ((5, 10.5), (5, 10)),      # User to Router
        ((5, 8.5), (5, 7.5)),      # Router to Decision
        ((4.2, 6.2), (2.8, 5)),    # Decision to Model
        ((5.8, 6.2), (7.2, 5)),    # Decision to Research
        ((2.25, 3), (4, 2)),       # Model to Output
        ((7.75, 3), (6, 2))        # Research to Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Labels for decision paths
    ax.text(3, 5.8, 'Model/\nArchitecture\nQuery', ha='center', va='center', fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax.text(7, 5.8, 'Research\nPlanning\nQuery', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def create_model_suggestion_flow(ax):
    """Create detailed model suggestion workflow."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_title('📊 Model Suggestion Workflow Detail', fontsize=14, fontweight='bold')
    
    # Workflow steps
    steps = [
        (5, 12.5, '📝 Analyze Properties\n& Task', '#E3F2FD'),
        (5, 11, '🔍 Generate Search\nQuery', '#E1F5FE'),
        (5, 9.5, '📚 Search ArXiv\nPapers', '#E0F2F1'),
        (5, 8, '✅ Validate Papers\n(NEW!)', '#FFF3E0'),
        (2.5, 6.5, '🔄 Search\nBackup', '#FFEBEE'),
        (5, 6.5, '💡 Suggest\nModels', '#E8F5E8'),
        (7.5, 6.5, '🔍 Critique\nResponse', '#F3E5F5'),
        (5, 5, '📋 Finalize\nResults', '#FFEBEE'),
        (7.5, 5, '🔄 Revise\nSuggestions', '#FFF9C4')
    ]
    
    # Draw boxes
    for x, y, text, color in steps:
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Decision diamonds
    decision1_points = np.array([[5, 7.5], [4.2, 7], [5, 6.5], [5.8, 7]])
    decision1 = patches.Polygon(decision1_points, closed=True, 
                               facecolor='#FFF9C4', edgecolor='black', linewidth=1)
    ax.add_patch(decision1)
    ax.text(5, 7, 'Papers\nOK?', ha='center', va='center', fontsize=7, fontweight='bold')
    
    decision2_points = np.array([[7.5, 4.2], [6.7, 3.7], [7.5, 3.2], [8.3, 3.7]])
    decision2 = patches.Polygon(decision2_points, closed=True,
                               facecolor='#FFF9C4', edgecolor='black', linewidth=1)
    ax.add_patch(decision2)
    ax.text(7.5, 3.7, 'Quality\nOK?', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Arrows - main flow
    main_arrows = [
        ((5, 12), (5, 11.5)),      # Properties to Query
        ((5, 10.5), (5, 10)),      # Query to ArXiv
        ((5, 9), (5, 8.5)),        # ArXiv to Validate
        ((5, 7.5), (5, 7.2)),      # Validate to Decision
        ((5, 6.8), (5, 7.1)),      # Decision to Models (continue)
        ((5, 6), (5, 5.5)),        # Models to Finalize
        ((6.7, 6.5), (6.7, 6.5)),  # Models to Critique
        ((7.5, 6), (7.5, 4.5))     # Critique to Decision2
    ]
    
    for start, end in main_arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    # Conditional arrows
    # Search backup
    ax.annotate('', xy=(3.3, 6.5), xytext=(4.2, 7),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))
    ax.text(3.5, 7.5, 'backup', ha='center', va='center', fontsize=7, color='orange')
    
    # Search new (loop back to query)
    ax.annotate('', xy=(3.5, 10.5), xytext=(4.2, 7),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='red', 
                             connectionstyle="arc3,rad=-0.3"))
    ax.text(2, 8.5, 'new\nsearch', ha='center', va='center', fontsize=7, color='red')
    
    # Revision loop
    ax.annotate('', xy=(6.5, 6), xytext=(7.5, 3.2),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='purple',
                             connectionstyle="arc3,rad=0.3"))
    ax.text(8.5, 4.5, 'revise', ha='center', va='center', fontsize=7, color='purple')
    
    # Final approval
    ax.annotate('', xy=(6.5, 5), xytext=(6.7, 3.7),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
    ax.text(6, 4, 'approve', ha='center', va='center', fontsize=7, color='green')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def create_research_planning_flow(ax):
    """Create detailed research planning workflow."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_title('🧪 Research Planning Workflow Detail', fontsize=14, fontweight='bold')
    
    # Workflow steps
    steps = [
        (5, 12.5, '🎯 Generate\nProblem', '#F3E5F5'),
        (5, 11, '✅ Validate\nProblem', '#E1F5FE'),
        (3, 9.5, '📥 Collect\nProblem', '#E8F5E8'),
        (7, 9.5, '🔄 Continue\nGeneration', '#FFEBEE'),
        (5, 8, '🎯 Select\nProblem', '#FFF3E0'),
        (5, 6.5, '📋 Create Research\nPlan', '#E0F2F1'),
        (5, 5, '🔍 Critique\nPlan', '#F3E5F5'),
        (5, 3.5, '✅ Finalize\nPlan', '#E8F5E8'),
        (2, 2, '🔄 Refine\nPlan', '#FFF9C4'),
        (8, 2, '🎯 Re-select\nProblem', '#FFEBEE')
    ]
    
    # Draw boxes
    for x, y, text, color in steps:
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Decision diamonds
    decision_points = [
        (5, 10.2, 'Valid?'),
        (5, 7.2, 'Enough?'),
        (5, 4.2, 'Quality\nOK?')
    ]
    
    for x, y, label in decision_points:
        diamond_points = np.array([[x, y+0.3], [x-0.6, y], [x, y-0.3], [x+0.6, y]])
        diamond = patches.Polygon(diamond_points, closed=True,
                                facecolor='#FFF9C4', edgecolor='black', linewidth=1)
        ax.add_patch(diamond)
        ax.text(x, y, label, ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Main flow arrows
    main_arrows = [
        ((5, 12), (5, 11.5)),      # Generate to Validate
        ((5, 10.5), (5, 10.5)),    # Validate to Decision
        ((4.4, 10.2), (3.8, 9.8)),# Decision to Collect (yes)
        ((5.6, 10.2), (6.2, 9.8)),# Decision to Continue (no)
        ((3, 9), (4, 8.3)),       # Collect to Decision2
        ((5, 7.5), (5, 7.1)),     # Decision2 to Select
        ((5, 7), (5, 6.8)),       # Select to Plan
        ((5, 6), (5, 5.5)),       # Plan to Critique
        ((5, 4.5), (5, 4.1)),     # Critique to Decision3
        ((5, 3.8), (5, 4.1)),     # Decision3 to Finalize (yes)
    ]
    
    for start, end in main_arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    # Loop arrows
    # Continue generation loop
    ax.annotate('', xy=(6, 12), xytext=(7, 9),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='orange',
                             connectionstyle="arc3,rad=0.4"))
    
    # Need more problems loop
    ax.annotate('', xy=(6, 12), xytext=(5.6, 7.2),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='green',
                             connectionstyle="arc3,rad=0.5"))
    
    # Refinement loops
    ax.annotate('', xy=(3.5, 6), xytext=(4.4, 4.2),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='purple'))
    
    ax.annotate('', xy=(6.5, 7.5), xytext=(5.6, 4.2),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
    
    # Labels
    ax.text(2.5, 10.5, 'yes', ha='center', va='center', fontsize=7, color='green')
    ax.text(6.5, 10.5, 'no', ha='center', va='center', fontsize=7, color='red')
    ax.text(3.5, 7.8, 'yes', ha='center', va='center', fontsize=7, color='green')
    ax.text(6.5, 7.8, 'no', ha='center', va='center', fontsize=7, color='orange')
    ax.text(3, 3.5, 'refine', ha='center', va='center', fontsize=7, color='purple')
    ax.text(7, 3.5, 're-select', ha='center', va='center', fontsize=7, color='red')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def create_router_decision_flow(ax):
    """Create router decision logic flow."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('🤖 Router Decision Logic', fontsize=14, fontweight='bold')
    
    # Input
    input_box = FancyBboxPatch((4, 8.5), 2, 1, boxstyle="round,pad=0.1",
                              facecolor='#E3F2FD', edgecolor='black', linewidth=1)
    ax.add_patch(input_box)
    ax.text(5, 9, 'User Query', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Analysis
    analysis_box = FancyBboxPatch((3.5, 7), 3, 1, boxstyle="round,pad=0.1",
                                 facecolor='#FFF3E0', edgecolor='black', linewidth=1)
    ax.add_patch(analysis_box)
    ax.text(5, 7.5, 'Semantic Analysis\nIntent Classification', ha='center', va='center', fontsize=9)
    
    # Decision criteria
    criteria = [
        (1.5, 5.5, 'Model/Architecture\nKeywords?\n\n• "model"\n• "architecture"\n• "algorithm"\n• "implement"', '#E8F5E8'),
        (8.5, 5.5, 'Research/Planning\nKeywords?\n\n• "research"\n• "plan"\n• "study"\n• "investigate"', '#F3E5F5')
    ]
    
    for x, y, text, color in criteria:
        box = FancyBboxPatch((x-1.3, y-1.3), 2.6, 2.6, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8)
    
    # Outputs
    outputs = [
        (1.5, 2, 'Model Suggestion\nWorkflow', '#E8F5E8'),
        (8.5, 2, 'Research Planning\nWorkflow', '#F3E5F5')
    ]
    
    for x, y, text, color in outputs:
        box = FancyBboxPatch((x-1, y-0.7), 2, 1.4, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrows = [
        ((5, 8.5), (5, 8)),        # Input to Analysis
        ((4, 7), (2.5, 6.5)),      # Analysis to Model criteria
        ((6, 7), (7.5, 6.5)),      # Analysis to Research criteria
        ((1.5, 4), (1.5, 2.7)),    # Model criteria to output
        ((8.5, 4), (8.5, 2.7))     # Research criteria to output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def create_state_management_flow(ax):
    """Create state management flow."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('🗂️ State Management Flow', fontsize=14, fontweight='bold')
    
    # State types
    states = [
        (2, 8.5, 'RouterState\n\n• messages\n• routing_decision\n• confidence\n• reasoning', '#E3F2FD'),
        (8, 8.5, 'ModelSuggestionState\n\n• detected_categories\n• arxiv_results\n• model_suggestions\n• validation_results', '#E8F5E8'),
        (5, 5.5, 'ResearchPlanningState\n\n• validated_problems\n• selected_problem\n• research_plan\n• critique_results', '#F3E5F5'),
        (5, 2, 'Final Output\n\n• JSON Results\n• Structured Analysis\n• Workflow Metadata', '#FFEBEE')
    ]
    
    for x, y, text, color in states:
        box = FancyBboxPatch((x-1.5, y-1.2), 3, 2.4, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8)
    
    # State transitions
    arrows = [
        ((2, 7.3), (7, 7.3)),      # Router to Model (horizontal)
        ((2, 7.3), (4, 6.7)),      # Router to Research
        ((8, 7.3), (6, 6.7)),      # Model to Research (if needed)
        ((7, 5.5), (6, 3.2)),      # Model to Output
        ((4, 4.3), (4.5, 3.2))     # Research to Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    # Labels
    ax.text(4.5, 7.8, 'Route to\nModel', ha='center', va='center', fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    ax.text(3, 6, 'Route to\nResearch', ha='center', va='center', fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def create_text_flow_summary():
    """Create a text-based flow summary for quick reference."""
    summary = """
🌟 ML RESEARCH ASSISTANT - WORKFLOW FLOW CHART SUMMARY
═══════════════════════════════════════════════════════

📊 MAIN SYSTEM FLOW:
User Input → Router → [Model Workflow OR Research Workflow] → JSON Output

🤖 ROUTER LOGIC:
1. Semantic analysis of user query
2. Keyword detection (model/architecture vs research/planning)
3. Confidence scoring and routing decision
4. Route to appropriate specialized workflow

📊 MODEL SUGGESTION WORKFLOW:
1. analyze_properties_and_task → Extract ML categories and requirements
2. generate_search_query → Create optimized ArXiv search terms
3. search_arxiv → Find and download relevant papers
4. validate_papers → NEW! LLM validation checkpoint with routing:
   - "continue" → Proceed with current papers
   - "search_backup" → Search for additional papers
   - "search_new" → Start fresh with new search query
5. suggest_models → Generate model recommendations with evidence
6. critique_response → Quality assessment and improvement suggestions
7. Decision: "finalize" (end) OR "revise" (loop back to suggest_models)

🧪 RESEARCH PLANNING WORKFLOW:
1. generate_problem → Create novel research questions
2. validate_problem → Web search validation of novelty/feasibility
3. Decision: "collect_problem" (accept) OR "continue_generation" (reject)
4. collect_problem → Store validated problems
5. Decision: "generate_problem" (need more) OR "select_problem" (enough)
6. select_problem → User selects from collected problems
7. create_research_plan → Generate comprehensive research plan
8. critique_plan → Quality assessment and improvement analysis
9. Decision paths:
   - "finalize_plan" → End with approved plan
   - "refine_plan" → Loop back to create_research_plan
   - "select_problem" → Choose different problem
   - "generate_problem" → Start over with new problems

🗂️ STATE MANAGEMENT:
- RouterState: routing decisions and confidence scores
- ModelSuggestionState: papers, models, validation results, critique data
- ResearchPlanningState: problems, plans, critique results, iterations
- Persistent state across nodes with error tracking and workflow metadata

🔄 KEY INNOVATIONS:
- Multi-workflow architecture with intelligent routing
- Paper validation checkpoint with adaptive search strategies
- Iterative critique and refinement for quality assurance
- State persistence across workflow transitions
- Comprehensive error handling and recovery

💡 CONDITIONAL LOGIC:
- Router uses semantic analysis for workflow selection
- Paper validation uses LLM assessment for search continuation
- Critique systems use quality thresholds for revision decisions
- Problem generation uses validation scores for acceptance
- Plan refinement uses issue severity for improvement paths
"""
    
    with open('workflow_flow_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("📝 Text flow summary saved to 'workflow_flow_summary.txt'")

if __name__ == "__main__":
    print("🎨 Generating comprehensive ML Research Assistant flow charts...")
    
    # Create visual flow chart
    create_comprehensive_flow_chart()
    
    # Create text summary
    create_text_flow_summary()
    
    print("✅ Flow charts generated successfully!")
    print("📊 Visual chart: 'ml_research_workflow_comprehensive.png'")
    print("📝 Text summary: 'workflow_flow_summary.txt'")
