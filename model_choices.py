import os
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel
from enum import Enum


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    QWEN = "qwen"


class LLMConfig(BaseModel):
    provider: ModelProvider
    model_name: str
    temperature: float = 1
    # max_tokens: int = 4096
    api_key: Optional[str] = None


class LLMClient:
    """Unified interface for calling different LLM APIs"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate API client based on provider"""
        if self.config.provider == ModelProvider.OPENAI:
            import openai
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            openai.api_key = api_key
            return openai
        
        elif self.config.provider == ModelProvider.ANTHROPIC:
            import anthropic
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            return anthropic.Anthropic(api_key=api_key)
        
        elif self.config.provider == ModelProvider.GOOGLE:
            import google.generativeai as genai
            api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            return genai
        
        elif self.config.provider == ModelProvider.QWEN:
            # Add Qwen initialization here
            # import qwen
            # api_key = self.config.api_key or os.getenv("QWEN_API_KEY")
            # return qwen.Client(api_key=api_key)
            raise NotImplementedError("Qwen provider not yet implemented")
        
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def call_llm(self, prompt: str, response_format: Optional[Any] = None) -> str:
        """
        Call the LLM with the given prompt and return the response.
        
        Args:
            prompt: The prompt to send to the LLM
            response_format: Optional Pydantic model for structured output
            
        Returns:
            The LLM's response as a string
        """
        try:
            if self.config.provider == ModelProvider.OPENAI:
                return self._call_openai(prompt, response_format)
            
            elif self.config.provider == ModelProvider.ANTHROPIC:
                return self._call_anthropic(prompt, response_format)
            
            elif self.config.provider == ModelProvider.GOOGLE:
                return self._call_google(prompt, response_format)
            
            elif self.config.provider == ModelProvider.QWEN:
                return self._call_qwen(prompt, response_format)
                
        except Exception as e:
            print(f"Error calling {self.config.provider}: {str(e)}")
            raise
    
    def _call_openai(self, prompt: str, response_format: Optional[Any] = None) -> str:
        """Call OpenAI API"""
        kwargs = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": 4096
        }
        
        # For structured outputs with Pydantic models, use the parse method
        if response_format:
            try:
                # Use the newer beta parse method for structured outputs
                response = self.client.beta.chat.completions.parse(
                    model=kwargs["model"],
                    messages=kwargs["messages"],
                    temperature=kwargs["temperature"],
                    max_tokens=kwargs["max_tokens"],
                    response_format=response_format
                )
                return response.choices[0].message.content
            except AttributeError:
                # Fallback: if beta.parse not available, use standard completion with JSON mode
                kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
        else:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, response_format: Optional[Any] = None) -> str:
        """Call Anthropic API"""
        # Note: Anthropic doesn't have native structured output like OpenAI
        # You'd need to add instructions to the prompt for JSON format
        tool_spec = {
        "name": "extract_answe_info",
        "description": "Extracts information from text.",
        "input_schema": response_format.model_json_schema() # Pydantic converts to JSON schema
                   }

        kwargs = {
            "model": self.config.model_name,
            "max_tokens": 4096,
            "temperature": self.config.temperature,
            "tools": [tool_spec],
            "messages": [{"role": "user", "content": prompt}]
        }

        # if response_format:
        #     # Add JSON instructions to the prompt
        #     #json_instruction = f"\n\nPlease respond in valid JSON format matching this schema: {response_format.model_json_schema()}"
        #     kwargs["messages"][0]["content"] = prompt 
        

      
        
        response = self.client.messages.create(**kwargs)
        print(f"ANTHROPIC RESPONSE:",response)
        
        # Extract answer from tool use block instead of text block
        for content_block in response.content:
            if content_block.type == 'tool_use':
                # Return the tool input as JSON string
                return json.dumps(content_block.input)
        
        # Fallback to text if no tool use found
        return response.content[0].text
    
    def _call_google(self, prompt: str, response_format: Optional[Any] = None) -> str:
        """Call Google Gemini API"""
        model = self.client.GenerativeModel(self.config.model_name)
        
        generation_config = {
            "temperature": self.config.temperature,
            "max_output_tokens": 4096
        }
        
        # Google doesn't have native structured output, add JSON instructions if needed
        if response_format:
            json_instruction = f"\n\nPlease respond in valid JSON format matching this schema: {response_format.model_json_schema()}"
            prompt = prompt + json_instruction
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    
    def _call_qwen(self, prompt: str, response_format: Optional[Any] = None) -> str:
        """Call Qwen API"""
        # Implement Qwen API call here
        raise NotImplementedError("Qwen provider not yet implemented")


def get_llm_client(provider: str, model_name: str, temperature: float = 1.0, api_key: Optional[str] = None) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: The LLM provider ("openai", "anthropic", "google", "qwen")
        model_name: The specific model to use (e.g., "gpt-4", "claude-3-opus-20240229")
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens in response
        api_key: Optional API key. If not provided, will use environment variables
        
    Returns:
        An initialized LLMClient
    """
    config = LLMConfig(
        provider=ModelProvider(provider),
        model_name=model_name,
        temperature=temperature,
        # max_tokens=max_tokens,
        api_key=api_key
    )
    return LLMClient(config)


if __name__ == "__main__":
    # Test the client
    print("Testing LLM Clients...")
    
    # Test OpenAI
    try:
        client = get_llm_client("openai", "gpt-4")
        print(f"✓ OpenAI client initialized")
    except Exception as e:
        print(f"✗ OpenAI error: {e}")
    
    # Test Anthropic
    try:
        client = get_llm_client("anthropic", "claude-3-5-sonnet-20241022")
        print(f"✓ Anthropic client initialized")
    except Exception as e:
        print(f"✗ Anthropic error: {e}")
    
    # Test Google
    try:
        client = get_llm_client("google", "gemini-pro")
        print(f"✓ Google client initialized")
    except Exception as e:
        print(f"✗ Google error: {e}")
