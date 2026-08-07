import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

logger = logging.getLogger("ocean_mind.mcp_client")
logger.setLevel(logging.INFO)

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        """Generate text response from LLM."""
        pass


class OpenAIStrategy(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        except ImportError:
            self.client = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized or missing API key.")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()


class AnthropicStrategy(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        except ImportError:
            self.client = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        if not self.client:
            raise RuntimeError("Anthropic client not initialized or missing API key.")
        
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text.strip()


class GeminiStrategy(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            self.genai = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        if not self.genai:
            raise RuntimeError("Google GenerativeAI package not installed or configured.")
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        model = self.genai.GenerativeModel(self.model)
        response = model.generate_content(
            full_prompt,
            generation_config=self.genai.types.GenerationConfig(temperature=temperature)
        )
        return response.text.strip()


class OllamaStrategy(BaseLLMClient):
    def __init__(self, host: Optional[str] = None, model: str = "llama3"):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        import requests
        url = f"{self.host.rstrip('/')}/api/generate"
        full_prompt = f"System: {system_prompt}\nUser: {prompt}" if system_prompt else prompt
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()


class MockLLMStrategy(BaseLLMClient):
    def __init__(self, response_map: Optional[Dict[str, str]] = None):
        self.response_map = response_map or {}
        self.default_response = "SELECT * FROM measurements LIMIT 10;"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        for key, resp in self.response_map.items():
            if key.lower() in prompt.lower():
                return resp
        return self.default_response


class MCPClient(BaseLLMClient):
    """
    Lightweight LLM wrapper supporting MCP server, primary providers with automatic fallback,
    and Strategy Pattern.
    """
    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.mcp_server_url = getattr(config, "MCP_SERVER_URL", os.environ.get("MCP_SERVER_URL"))
        self.provider = getattr(config, "LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "mock")).lower()
        self.api_key = getattr(config, "LLM_API_KEY", os.environ.get("LLM_API_KEY"))
        self.model_name = getattr(config, "LLM_MODEL", os.environ.get("LLM_MODEL", "gpt-4o"))
        
        self.strategy = self._initialize_strategy()

    def _initialize_strategy(self) -> BaseLLMClient:
        if self.mcp_server_url:
            try:
                logger.info(f"Connecting to MCP server at {self.mcp_server_url}")
            except Exception as e:
                logger.warning(f"MCP server unreachable ({e}), falling back to direct provider {self.provider}")

        if self.provider == "openai":
            return OpenAIStrategy(api_key=self.api_key, model=self.model_name)
        elif self.provider == "anthropic":
            return AnthropicStrategy(api_key=self.api_key, model=self.model_name)
        elif self.provider == "gemini":
            return GeminiStrategy(api_key=self.api_key, model=self.model_name)
        elif self.provider == "ollama":
            return OllamaStrategy(model=self.model_name)
        else:
            return MockLLMStrategy()

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        try:
            return self.strategy.generate(prompt, system_prompt=system_prompt, temperature=temperature)
        except Exception as e:
            logger.error(f"Primary strategy {self.provider} failed: {e}. Falling back to Mock/Default.")
            fallback = MockLLMStrategy()
            return fallback.generate(prompt, system_prompt=system_prompt, temperature=temperature)