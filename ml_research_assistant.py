#!/usr/bin/env python3
"""
ML Research Assistant - ChatGPT Wrapper for Machine Learning Research
==================================================================

A tool that leverages ChatGPT to generate comprehensive research plans
for machine learning topics and projects.

Usage:
    python ml_research_assistant.py --topic "your ML topic"
    python ml_research_assistant.py --interactive
    python ml_research_assistant.py --project "project description"

Requirements:
    - OpenAI API key (set as environment variable OPENAI_API_KEY)
    - openai library (pip install openai)
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional
import openai
from openai import OpenAI


class MLResearchAssistant:
    """ChatGPT wrapper for generating ML research plans."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None, max_tokens: Optional[int] = None):
        """Initialize the research assistant with LiteLLM proxy configuration."""
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url or self._get_base_url()
        self.model = model or self._get_model()
        self.max_tokens = max_tokens or self._get_max_tokens()
        
        if not self.api_key:
            raise ValueError("API key not found. Check env.example file or set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client with custom base URL for LiteLLM proxy
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.research_template = self._load_research_template()
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources in order of preference."""
        # 1. Check environment variable first
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # 2. Check .env file
        env_file = '.env'
        if os.path.exists(env_file):
            api_key = self._read_config_from_file(env_file, 'OPENAI_API_KEY')
            if api_key:
                return api_key
        
        # 3. Check env.example file
        env_example_file = 'env.example'
        if os.path.exists(env_example_file):
            api_key = self._read_config_from_file(env_example_file, 'OPENAI_API_KEY')
            if api_key:
                return api_key
        
        return None
    
    def _get_base_url(self) -> str:
        """Get base URL from config files or use default."""
        # Check environment variable
        base_url = os.getenv('BASE_URL')
        if base_url:
            return base_url
        
        # Check config files
        for filename in ['.env', 'env.example']:
            if os.path.exists(filename):
                base_url = self._read_config_from_file(filename, 'BASE_URL')
                if base_url:
                    return base_url
        
        # Default LiteLLM proxy URL
        return 'https://agents.aetherraid.dev'
    
    def _get_model(self) -> str:
        """Get model from config files or use default."""
        # Check environment variable
        model = os.getenv('DEFAULT_MODEL')
        if model:
            return model
        
        # Check config files
        for filename in ['.env', 'env.example']:
            if os.path.exists(filename):
                model = self._read_config_from_file(filename, 'DEFAULT_MODEL')
                if model:
                    return model
        
        # Default Gemini model
        return 'gemini/gemini-2.5-flash'
    
    def _get_max_tokens(self) -> int:
        """Get max tokens from config files or use default."""
        # Check environment variable
        max_tokens = os.getenv('MAX_TOKENS')
        if max_tokens:
            try:
                return int(max_tokens)
            except ValueError:
                pass
        
        # Check config files
        for filename in ['.env', 'env.example']:
            if os.path.exists(filename):
                max_tokens = self._read_config_from_file(filename, 'MAX_TOKENS')
                if max_tokens:
                    try:
                        return int(max_tokens)
                    except ValueError:
                        pass
        
        # Default for comprehensive research plans
        return 8000
    
    def _read_config_from_file(self, filename: str, key: str) -> Optional[str]:
        """Read configuration value from environment file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f'{key}='):
                        # Handle both quoted and unquoted values
                        value = line.split('=', 1)[1]
                        # Remove quotes if present
                        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                            value = value[1:-1]
                        if value and value not in ['your-api-key-here', 'your-base-url-here']:
                            return value
        except Exception:
            pass
        return None
    
    def _load_research_template(self) -> str:
        """Load the task decomposition and model architecture analysis template."""
        return """
You are an expert ML researcher specializing in task decomposition and model architecture analysis. Given a specific machine learning task, provide a comprehensive analysis that decomposes the task into fundamental properties and identifies suitable model architectures.

Structure your response as follows:

# Task Decomposition & Architecture Analysis: {topic}



## 2. Task Property Decomposition Framework
Systematically analyze the given task along multiple dimensions:

### 2.1 Temporal Properties
- **Time-invariant vs Time-variant**: Does the task require understanding temporal patterns?
- **Lag-invariant vs Lag-sensitive**: Is the timing/delay between events important?
- **Sequential vs Non-sequential**: Does order of inputs matter?
- **Stationary vs Non-stationary**: Do statistical properties change over time?
- **Short-term vs Long-term dependencies**: What temporal range is relevant?

### 2.2 Input/Output Properties
- **Fixed-length vs Variable-length**: Are input/output sequences of consistent size?
- **Uni-modal vs Multi-modal**: Single or multiple types of input data?
- **Structured vs Unstructured**: Is there inherent structure in the data?
- **Dense vs Sparse**: Data density characteristics
- **Continuous vs Discrete**: Nature of the data values

### 2.3 Learning Properties
- **Supervised vs Unsupervised vs Semi-supervised**: Label availability
- **Online vs Batch learning**: Learning paradigm requirements
- **Transfer learning potential**: Can pre-trained models be leveraged?
- **Few-shot vs Many-shot**: Data availability constraints
- **Incremental vs Static**: Does the model need to adapt over time?

### 2.4 Computational Properties
- **Real-time vs Offline**: Latency requirements
- **Memory constraints**: Working memory vs long-term memory needs
- **Scalability requirements**: How does complexity grow with data?
- **Interpretability needs**: Black-box vs explainable requirements
- **Resource constraints**: Computational budget limitations

### 2.5 Domain-Specific Properties
- **Spatial locality**: Are nearby elements more related?
- **Hierarchical structure**: Multi-level patterns or representations
- **Invariance requirements**: Translation, rotation, scale invariance needs
- **Noise tolerance**: Robustness requirements
- **Generalization scope**: Domain transfer expectations

## 3. Property-Architecture Mapping Matrix
Create a systematic mapping between identified task properties and suitable architectures:

### 3.1 Temporal Processing Architectures
- **Recurrent Neural Networks (RNNs)**
  - LSTM/GRU for long-term dependencies
  - Vanilla RNN for simple temporal patterns
  - Bidirectional variants for full sequence context
- **Transformer-based Models**
  - Self-attention for parallel temporal processing
  - Positional encoding for sequence awareness
  - Temporal transformers for time series
- **Convolutional Temporal Models**
  - 1D CNNs for local temporal patterns
  - Dilated convolutions for multi-scale temporal features
  - Temporal convolutional networks (TCNs)

### 3.2 Variable-Length Processing Architectures
- **Sequence-to-Sequence Models**
  - Encoder-decoder architectures
  - Attention mechanisms for alignment
  - Copy mechanisms for variable output
- **Dynamic Neural Networks**
  - Adaptive computation time models
  - Dynamic routing networks
  - Variable-depth networks

### 3.3 Multi-Modal and Structured Data Architectures
- **Graph Neural Networks (GNNs)**
  - For structured/relational data
  - Graph attention networks
  - Temporal graph networks
- **Fusion Architectures**
  - Early vs late fusion strategies
  - Cross-modal attention
  - Multi-stream architectures

### 3.4 Memory and Context Architectures
- **Memory-Augmented Networks**
  - Neural Turing Machines
  - Differentiable Neural Computers
  - Memory networks
- **Hierarchical Models**
  - Multi-scale processing
  - Pyramidal networks
  - Hierarchical attention

## 4. Architecture Selection Methodology
Provide a systematic approach for choosing optimal architectures:

### 4.1 Property-Driven Selection Process
1. **Property Identification**: Map task to property checklist
2. **Constraint Analysis**: Identify hard constraints (computational, data, time)
3. **Architecture Filtering**: Eliminate incompatible architectures
4. **Multi-Criteria Evaluation**: Score remaining candidates
5. **Ensemble Consideration**: Combine complementary architectures

### 4.2 Decision Trees and Flowcharts
- Create decision trees for architecture selection
- Provide flowcharts for systematic evaluation
- Include trade-off analysis frameworks

### 4.3 Evaluation Metrics Framework
- Task-specific performance metrics
- Computational efficiency metrics
- Generalization capability assessment
- Robustness evaluation criteria

## 5. Literature Review and Existing Approaches
Survey current research on task decomposition and architecture design:

### 5.1 Theoretical Foundations
- Key papers on task decomposition in ML
- Universal approximation theorems for different architectures
- Inductive biases and their relationship to task properties
- Meta-learning approaches for architecture selection

### 5.2 Empirical Studies
- Comparative studies of architectures on similar tasks
- Ablation studies revealing important properties
- Architecture search methodologies (NAS)
- Benchmark datasets and their property profiles

### 5.3 Recent Advances
- Automated architecture design
- Property-aware neural architecture search
- Foundation models and their property coverage
- Hybrid and modular architectures

## 6. Practical Implementation Framework

### 6.1 Task Analysis Pipeline
```
Input Task → Property Extraction → Architecture Mapping → Selection Algorithm → Implementation Plan
```

### 6.2 Tools and Methodologies
- Property profiling tools and checklists
- Architecture comparison frameworks
- Implementation templates for common patterns
- Evaluation harnesses for systematic comparison

### 6.3 Code Architecture Patterns
- Modular design for property-based selection
- Abstract base classes for different architecture families
- Configuration-driven architecture instantiation
- Automated hyperparameter optimization

## 7. Case Studies and Examples
Provide concrete examples of the decomposition process:

### 7.1 Example Task Decompositions
- Time series forecasting task breakdown
- Natural language processing task analysis
- Computer vision task property mapping
- Multi-modal learning scenario decomposition

### 7.2 Architecture Selection Examples
- Step-by-step selection process for each case study
- Justification for chosen architectures
- Alternative architectures and trade-offs
- Performance comparison results

## 8. Experimental Design and Validation

### 8.1 Systematic Evaluation Protocol
- Controlled experiments across property dimensions
- Ablation studies for individual properties
- Cross-validation strategies
- Statistical significance testing

### 8.2 Benchmarking Framework
- Property-aware benchmark creation
- Standardized evaluation metrics
- Baseline architecture comparisons
- Reproducibility guidelines

## 9. Future Research Directions

### 9.1 Open Research Questions
- Automated property identification from task descriptions
- Dynamic architecture adaptation based on observed properties
- Property-architecture interaction effects
- Theoretical bounds on architecture-property matching

### 9.2 Emerging Trends
- Neural architecture search guided by task properties
- Meta-learning for architecture selection
- Compositional architectures
- Foundation model adaptation strategies

## 10. Tools and Resources

### 10.1 Software Tools
- Property analysis libraries
- Architecture search frameworks
- Visualization tools for property-architecture mapping
- Benchmark datasets organized by properties

### 10.2 Development Guidelines
- Best practices for implementing property-aware systems
- Common pitfalls and how to avoid them
- Performance optimization strategies
- Debugging and validation approaches

Provide specific, actionable analysis that researchers can use to systematically approach any ML task through property decomposition and informed architecture selection.
"""

    def generate_research_plan(self, topic: str) -> Dict:
        """Generate a comprehensive research plan for the given ML topic."""
        
        prompt = f"""
{self.research_template.format(topic=topic)}

Topic/Project: {topic}

Please provide a detailed, well-structured research plan that covers all aspects:
- Theoretical foundations and mathematical formulations
- Practical implementation with code examples and frameworks
- Real-world applications and industry use cases
- Deployment strategies and system architecture

The plan should be comprehensive enough for both academic research and practical implementation.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert machine learning researcher and academic advisor with deep knowledge across all ML domains."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            plan_content = response.choices[0].message.content
            
            # Create structured response
            research_plan = {
                "topic": topic,
                "generated_at": datetime.now().isoformat(),
                "content": plan_content,
                "metadata": {
                    "model": self.model,
                    "base_url": self.base_url,
                    "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0,
                    "cost_estimate": self._estimate_cost(response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0)
                }
            }
            
            return research_plan
            
        except Exception as e:
            raise Exception(f"Error generating research plan: {str(e)}")
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate API cost based on token usage."""
        if tokens == 0:
            return 0.0
        
        # Generic pricing estimate (actual costs may vary by model and provider)
        if 'gemini' in self.model.lower():
            # Gemini pricing (approximate)
            cost_per_1k_tokens = 0.001  # Much cheaper than GPT-4
        elif 'gpt-4' in self.model.lower():
            # GPT-4 pricing (approximate)
            cost_per_1k_tokens = 0.03
        elif 'gpt-3.5' in self.model.lower():
            # GPT-3.5 pricing (approximate)
            cost_per_1k_tokens = 0.002
        else:
            # Default estimate
            cost_per_1k_tokens = 0.01
        
        return (tokens / 1000) * cost_per_1k_tokens
    
    def save_research_plan(self, research_plan: Dict, filename: Optional[str] = None) -> str:
        """Save the research plan to a file."""
        if not filename:
            safe_topic = "".join(c for c in research_plan["topic"] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_topic = safe_topic.replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_plan_{safe_topic}_{timestamp}.md"
        
        # Save as markdown
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# ML Research Plan\n")
            f.write(f"**Generated**: {research_plan['generated_at']}\n")
            f.write(f"**Topic**: {research_plan['topic']}\n\n")
            f.write("---\n\n")
            f.write(research_plan['content'])
            f.write(f"\n\n---\n")
            f.write(f"*Generated using ML Research Assistant*\n")
            f.write(f"*Tokens used: {research_plan['metadata']['tokens_used']}*\n")
            f.write(f"*Estimated cost: ${research_plan['metadata']['cost_estimate']:.4f}*\n")
        
        # Also save as JSON for programmatic access
        json_filename = filename.replace('.md', '.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(research_plan, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def interactive_mode(self):
        """Run the assistant in interactive mode."""
        print("🔬 ML Research Assistant - Interactive Mode")
        print("=" * 50)
        print("Type 'exit' to quit, 'help' for commands")
        
        while True:
            try:
                user_input = input("\n📝 Enter your ML research topic or question: ").strip()
                
                if user_input.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif not user_input:
                    continue
                
                print(f"\n🚀 Generating comprehensive research plan for: '{user_input}'...")
                print("⏳ This may take 30-60 seconds...")
                
                # Generate research plan
                research_plan = self.generate_research_plan(user_input)
                
                # Display the plan
                print("\n" + "=" * 80)
                print(research_plan['content'])
                print("=" * 80)
                
                # Ask if user wants to save
                save_choice = input("\n💾 Save this research plan? (y/n) [default: y]: ").strip().lower()
                if save_choice != 'n':
                    filename = self.save_research_plan(research_plan)
                    print(f"✅ Research plan saved to: {filename}")
                    print(f"📊 Tokens used: {research_plan['metadata']['tokens_used']}")
                    print(f"💰 Estimated cost: ${research_plan['metadata']['cost_estimate']:.4f}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
🔬 ML Task Decomposition & Architecture Analysis:

Available commands:
- Enter any ML task or problem to generate a decomposition analysis
- 'exit' - Quit the application
- 'help' - Show this help message

Examples:
- "Time series forecasting for stock prices"
- "Real-time object detection in autonomous vehicles"
- "Multi-modal sentiment analysis from text and images"
- "Variable-length sequence classification"
- "Recommendation system with sparse user interactions"

The assistant will generate a comprehensive analysis including:
✓ Task property decomposition (temporal, input/output, learning, computational)
✓ Property-architecture mapping matrix
✓ Architecture selection methodology
✓ Literature review of relevant approaches
✓ Implementation framework and tools
✓ Case studies and examples
✓ Experimental design guidelines
✓ Future research directions
"""
        print(help_text)


def main():
    """Main function to run the ML Research Assistant."""
    parser = argparse.ArgumentParser(
        description="ML Research Assistant - Task Decomposition & Architecture Analysis tool for generating ML research plans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml_research_assistant.py --topic "Variable-length sequence classification with temporal dependencies"
  python ml_research_assistant.py --interactive
  python ml_research_assistant.py --project "Multi-modal sentiment analysis with text and images"
        """
    )
    
    parser.add_argument(
        '--topic', '-t',
        type=str,
        help='ML task or problem to analyze through property decomposition'
    )
    
    parser.add_argument(
        '--project', '-p',
        type=str,
        help='ML project description to decompose and analyze for architecture selection'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output filename for the research plan'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for LiteLLM proxy (or set OPENAI_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model to use (default: gemini/gemini-2.5-flash or from config)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        help='Base URL for LiteLLM proxy (default: https://agents.aetherraid.dev or from config)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Maximum tokens for response (default: 8000 or from config)'
    )
    
    args = parser.parse_args()
    
    # Check if API key is available from any source
    api_key_available = (
        args.api_key or 
        os.getenv('OPENAI_API_KEY') or 
        os.path.exists('.env') or 
        os.path.exists('env.example')
    )
    
    if not api_key_available:
        print("❌ Error: OpenAI API key required!")
        print("You can provide the API key in several ways:")
        print("1. Command line: --api-key your-key-here")
        print("2. Environment variable: $env:OPENAI_API_KEY='your-key-here'")
        print("3. Create .env file with: OPENAI_API_KEY=your-key-here")
        print("4. Update env.example file with your actual API key")
        print("\nTo get an API key:")
        print("1. Go to https://platform.openai.com/api-keys")
        print("2. Create a new secret key")
        print("3. Add it to env.example file or set as environment variable")
        sys.exit(1)
    
    try:
        # Initialize the assistant with custom configuration
        assistant = MLResearchAssistant(
            api_key=getattr(args, 'api_key', None),
            model=getattr(args, 'model', None),
            base_url=getattr(args, 'base_url', None),
            max_tokens=getattr(args, 'max_tokens', None)
        )
        
        if args.interactive:
            # Run interactive mode
            assistant.interactive_mode()
        
        elif args.topic or args.project:
            # Generate research plan for specific topic/project
            topic = args.topic or args.project
            
            print(f"🚀 Generating comprehensive research plan for: '{topic}'")
            print("⏳ This may take 30-60 seconds...")
            
            research_plan = assistant.generate_research_plan(topic)
            
            # Display the plan
            print("\n" + "=" * 80)
            print(research_plan['content'])
            print("=" * 80)
            
            # Save the plan
            filename = assistant.save_research_plan(research_plan, args.output)
            print(f"\n✅ Research plan saved to: {filename}")
            print(f"📊 Tokens used: {research_plan['metadata']['tokens_used']}")
            print(f"💰 Estimated cost: ${research_plan['metadata']['cost_estimate']:.4f}")
        
        else:
            # No arguments provided, show help and run interactive mode
            parser.print_help()
            print("\n🔬 Starting interactive mode...\n")
            assistant.interactive_mode()
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
