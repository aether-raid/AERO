# 🎨 How to Use the Mermaid Visualizations

## 📁 Generated Files

Your `diagrams/` folder now contains:

1. **router_workflow.mmd** - Router and workflow decision logic
2. **model_suggestion_workflow.mmd** - Model recommendation pipeline  
3. **research_planning_workflow.mmd** - Iterative research planning
4. **complete_system_overview.mmd** - High-level system architecture
5. **state_flow_diagram.mmd** - Data structures and state management
6. **conditional_logic_diagram.mmd** - Decision points and logic
7. **workflow_viewer.html** - Standalone HTML viewer
8. **README_integration.md** - Ready-to-use README sections

## 🌐 Viewing Options

### Option 1: Mermaid Live Editor (Recommended)
1. Go to https://mermaid.live
2. Copy content from any `.mmd` file
3. Paste into the editor
4. Instantly see the visual diagram
5. Export as PNG/SVG if needed

### Option 2: Standalone HTML Viewer
1. Open `diagrams/workflow_viewer.html` in your browser ('start diagrams/workflow_viewer.html)
2. View all diagrams in one page
3. Use navigation to jump between sections

### Option 3: GitHub Integration
1. Copy content from `README_integration.md`
2. Add to your project's README.md
3. GitHub automatically renders mermaid diagrams

### Option 4: VS Code (with Extension)
1. Install "Mermaid Markdown Syntax Highlighting" extension
2. Open any `.mmd` file in VS Code
3. Use preview pane to see rendered diagram

## 🎯 Recommended Workflow

1. **Quick viewing**: Use mermaid.live for immediate visualization
2. **Documentation**: Copy sections from README_integration.md
3. **Presentations**: Export PNG/SVG from mermaid.live
4. **Development**: Use VS Code extension for editing

## 📝 Example: Using with Mermaid Live

1. Open https://mermaid.live
2. Copy this content from `complete_system_overview.mmd`:

```
graph TB
    subgraph "🌟 ML Researcher LangGraph System"
        A[👤 User Query<br/>Research Question] --> B[🤖 Intelligent Router<br/>Semantic Analysis]
        // ... rest of the diagram
    end
```

3. Paste into mermaid.live
4. See beautiful rendered diagram!
5. Export as needed

## 🔧 Customization

All `.mmd` files are text-based and fully editable:
- Modify colors by changing `classDef` statements
- Add/remove nodes by editing the graph structure  
- Update labels by changing text in brackets
- Adjust styling with CSS-like syntax

## 💡 Tips

- **Performance**: Large diagrams may take a moment to render
- **Mobile**: Some diagrams are better viewed on desktop
- **Export**: Use SVG for scalable graphics, PNG for presentations
- **Sharing**: Mermaid.live provides shareable URLs for diagrams
