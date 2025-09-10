# ML Researcher LangGraph - Frontend

A modern web frontend for the ML Researcher LangGraph system, providing an intuitive interface for AI-powered research assistance with multi-workflow orchestration.

## 🌟 Features

### 🎨 **Modern Web Interface**
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Analysis**: Live progress tracking and results streaming
- **Interactive Workflow Visualization**: Visual representation of system architecture
- **Multiple Operation Modes**: API mode (with backend) and Demo mode (standalone)

### 🚀 **Dual Operation Modes**

#### **API Mode** (Recommended)
- Full integration with LangGraph backend
- Real-time analysis with actual AI models
- Complete workflow orchestration
- Persistent history and result storage

#### **Demo Mode** (Standalone)
- Works without backend setup
- Simulated analysis results
- Educational and demonstration purposes
- Local browser storage

### 🔧 **User Experience Features**
- **Example Prompts**: Quick-start templates for different use cases
- **Tabbed Results**: Organized display of analysis outputs
- **Export Options**: JSON, text, and clipboard export
- **Analysis History**: Persistent local and server-side history
- **Connection Status**: Real-time backend connectivity indicator

## 📁 Files Overview

```
├── app.py                      # FastAPI backend server
├── frontend.html               # Standalone web frontend
├── start_backend.py           # Backend startup script
├── requirements_frontend.txt   # Frontend dependencies
└── README_FRONTEND.md         # This file
```

## 🚀 Quick Start

### Option 1: Full Setup (API Mode)

1. **Install Dependencies**
   ```powershell
   pip install -r requirements_frontend.txt
   ```

2. **Start Backend Server**
   ```powershell
   python start_backend.py
   ```

3. **Open Frontend**
   - Visit `http://localhost:8000` (embedded frontend)
   - Or open `frontend.html` in your browser

### Option 2: Demo Mode (No Backend Required)

1. **Open Standalone Frontend**
   ```powershell
   start frontend.html  # Windows
   # or double-click frontend.html
   ```

2. **The frontend will automatically detect no backend and switch to demo mode**

## 🔧 Configuration

### Backend Settings
- **Default URL**: `http://localhost:8000`
- **Port**: 8000 (configurable in `start_backend.py`)
- **CORS**: Enabled for all origins (configure in `app.py` for production)

### Frontend Settings
- **Connection Testing**: Automatic backend detection
- **Local Storage**: History and preferences saved locally
- **Responsive Breakpoints**: Mobile-first design

## 📊 API Endpoints

The backend provides the following REST API endpoints:

### **Core Analysis**
- `POST /analyze` - Analyze research task
- `POST /analyze/stream` - Stream analysis results
- `GET /health` - Backend health check

### **History Management**
- `GET /history` - Get analysis history
- `GET /history/{id}` - Get specific analysis

### **Frontend**
- `GET /` - Serve embedded frontend
- `GET /docs` - Swagger API documentation

## 🎯 Usage Examples

### Model Recommendation
```
What's the best model for image classification with limited data?
```

### Research Planning
```
Generate a research plan for adversarial robustness in deep learning
```

### Technical Questions
```
How to optimize transformer models for mobile deployment?
```

## 🔍 Workflow Types

### 🤖 **Model Suggestion Workflow**
- Task requirement analysis
- arXiv paper search and analysis
- Model architecture recommendations
- Implementation guidance
- AI quality critique and refinement

### 📋 **Research Planning Workflow**
- Research problem generation
- Web search validation
- Comprehensive plan creation
- Multi-dimensional quality assessment
- Iterative refinement process

### 💬 **Direct LLM Workflow**
- General query processing
- Direct AI responses
- No specialized workflow routing

## 🎨 Frontend Architecture

### **Technology Stack**
- **Framework**: Vue.js 3 (CDN)
- **Styling**: Tailwind CSS
- **Icons**: Font Awesome
- **HTTP Client**: Axios
- **State Management**: Vue Composition API

### **Components Structure**
- **Header**: Branding, connection status, settings
- **Workflow Overview**: Visual system architecture
- **Input Section**: Prompt entry, examples, options
- **Progress Display**: Real-time analysis logging
- **Results Tabs**: Organized output display
- **History Management**: Local and server-side history

### **Responsive Design**
- **Mobile First**: Optimized for mobile devices
- **Tablet Support**: Adaptive layout for tablets
- **Desktop Enhanced**: Full feature set on desktop
- **Print Friendly**: Clean printing layouts

## 🔐 Security Considerations

### **Development Setup**
- CORS enabled for all origins
- No authentication required
- Local file system access for exports

### **Production Recommendations**
- Configure specific CORS origins
- Add authentication middleware
- Use HTTPS for all communications
- Implement rate limiting
- Add input validation and sanitization

## 🐛 Troubleshooting

### **Common Issues**

#### **Backend Connection Failed**
- Verify backend is running: `python start_backend.py`
- Check firewall settings
- Confirm port 8000 is available
- Test with `curl http://localhost:8000/health`

#### **Analysis Not Working**
- Check browser console for errors
- Verify ML Researcher dependencies are installed
- Ensure environment variables are configured
- Try demo mode for troubleshooting

#### **Frontend Not Loading**
- Use modern browser (Chrome, Firefox, Safari, Edge)
- Check JavaScript is enabled
- Clear browser cache
- Try incognito/private mode

### **Debug Mode**
Enable verbose logging by:
1. Open browser developer tools (F12)
2. Check console tab for errors
3. Use network tab to monitor API calls
4. Enable verbose mode in frontend settings

## 🔄 Development

### **Backend Development**
```powershell
# Install in development mode
pip install -e .

# Start with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### **Frontend Development**
- Edit `frontend.html` directly
- Use browser developer tools
- Test responsive design with device simulation
- Validate with different browsers

### **Adding New Features**
1. **Backend**: Add endpoints in `app.py`
2. **Frontend**: Add Vue components and methods
3. **Testing**: Test both API and demo modes
4. **Documentation**: Update this README

## 📈 Performance Optimization

### **Backend**
- Use async/await for all operations
- Implement response caching
- Add connection pooling
- Monitor memory usage

### **Frontend**
- Lazy load heavy components
- Implement virtual scrolling for large lists
- Use component key for efficient re-rendering
- Optimize bundle size with tree shaking

## 📚 Additional Resources

### **Documentation**
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Vue.js 3 Guide](https://vuejs.org/guide/)

### **Examples**
- Model recommendation prompts
- Research planning templates
- API integration examples
- Custom workflow development

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add/update frontend components
4. Test both API and demo modes
5. Update documentation
6. Submit pull request

## 📄 License

This frontend is part of the ML Researcher LangGraph project and follows the same MIT License.

---

**🚀 Ready to start?** 
- For full experience: `python start_backend.py` then visit `http://localhost:8000`
- For quick demo: Open `frontend.html` in your browser
