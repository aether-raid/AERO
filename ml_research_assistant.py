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
        """Load the comprehensive research plan template for ML topics."""
        return """
You are an expert ML researcher and academic advisor. Generate a comprehensive research plan for the given machine learning topic or project that covers all aspects from theory to deployment.

Structure your response as follows:

# Research Plan: {topic}

## 1. Executive Summary
- Brief overview of the research area
- Key objectives and research questions
- Expected outcomes and impact
- Theoretical and practical contributions

## 2. Theoretical Foundations
- Mathematical formulations and core concepts
- Algorithmic principles and innovations
- Theoretical analysis and proofs (where applicable)
- Connections to existing theory

## 3. Background & Literature Review
- Current state of the field
- Key papers and researchers to study (provide specific paper titles and authors)
- Identified research gaps and opportunities
- Relevant datasets and benchmarks
- Comparison with existing approaches

## 4. Practical Implementation
- Programming languages and frameworks to use
- Code architecture and design patterns
- Key algorithms to implement
- Development environment setup
- Version control and collaboration tools

## 5. Real-World Applications
- Industry use cases and applications
- Business value and impact assessment
- Target users and stakeholders
- Market analysis and competitive landscape
- Ethical considerations and limitations

## 6. System Architecture & Deployment
- System design and architecture
- Scalability considerations
- Performance optimization strategies
- Deployment platforms and infrastructure
- Monitoring and maintenance plans
- API design and integration

## 7. Research Methodology & Experimentation
- Experimental design and validation methods
- Evaluation metrics and benchmarks
- Data collection and preprocessing
- Statistical analysis approach
- Reproducibility considerations

## 8. Technical Requirements
- Required skills and knowledge areas
- Software tools and frameworks
- Hardware requirements and computational resources
- Data requirements and sources
- Budget and resource planning

## 9. Implementation Timeline
### Phase 1: Foundation & Theory (Weeks 1-4)
- Literature review and theoretical study
- Mathematical formulation and analysis
- Environment setup and tool familiarization

### Phase 2: Prototype Development (Weeks 5-8)
- Core algorithm implementation
- Initial proof of concept
- Basic testing and validation

### Phase 3: Full Implementation (Weeks 9-12)
- Complete system development
- Advanced features and optimization
- Comprehensive testing

### Phase 4: Evaluation & Validation (Weeks 13-16)
- Extensive experimentation
- Performance evaluation and comparison
- Results analysis and interpretation

### Phase 5: Deployment & Documentation (Weeks 17-20)
- System deployment and integration
- Documentation and user guides
- Paper writing and presentation preparation

## 10. Risk Assessment & Mitigation
- Technical challenges and solutions
- Resource limitations and alternatives
- Timeline risks and contingency plans
- Data availability and quality issues

## 11. Success Metrics & Evaluation
- Quantitative evaluation criteria
- Qualitative assessment methods
- Academic publication goals
- Industry adoption metrics
- Long-term impact assessment

## 12. Resources & Next Steps
- Essential papers and tutorials to study
- Online courses and learning resources
- Relevant conferences and journals
- Open-source implementations to examine
- Community and collaboration opportunities

Provide specific, actionable recommendations throughout the plan that cover theoretical understanding, practical implementation, real-world applications, and deployment strategies.

## 8. Resources & References
- Essential papers to read
- Useful tutorials and courses
- Relevant conferences and journals
- Open-source implementations

Provide specific, actionable recommendations throughout the plan.
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
🔬 ML Research Assistant Commands:

Available commands:
- Enter any ML topic or research question to generate a research plan
- 'exit' - Quit the application
- 'help' - Show this help message

Examples:
- "Transformer models for computer vision"
- "Federated learning privacy preservation"
- "Graph neural networks for drug discovery"
- "Self-supervised learning for time series"

The assistant will generate a comprehensive research plan including:
✓ Theoretical foundations and mathematical formulations
✓ Practical implementation with code examples
✓ Real-world applications and use cases
✓ Deployment strategies and system architecture
✓ Literature review guidance
✓ Experimental design and evaluation
✓ Timeline and milestones
✓ Technical requirements
✓ Success metrics
"""
        print(help_text)


def main():
    """Main function to run the ML Research Assistant."""
    parser = argparse.ArgumentParser(
        description="ML Research Assistant - ChatGPT wrapper for generating ML research plans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml_research_assistant.py --topic "Transformer models for time series forecasting"
  python ml_research_assistant.py --interactive
  python ml_research_assistant.py --project "Build a recommendation system using graph neural networks"
        """
    )
    
    parser.add_argument(
        '--topic', '-t',
        type=str,
        help='ML research topic to generate a plan for'
    )
    
    parser.add_argument(
        '--project', '-p',
        type=str,
        help='ML project description to generate a research plan for'
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
