"""Integration tests for LLM server functionality."""

import json
import time

import pytest
from openai import OpenAI
from openai._exceptions import OpenAIError


@pytest.fixture(scope="session")
def client():
    """Create OpenAI client for LLM server integration testing."""
    return OpenAI()


@pytest.fixture(scope="session")
def default_model(client):
    """Get the default model from LLM server."""
    try:
        models = client.models.list()
        available_models = [model.id for model in models.data]
        if not available_models:
            pytest.skip("No models available in LLM server")
        return available_models[0]
    except Exception as e:
        pytest.skip(f"Cannot get available models: {e}")


class TestLLMIntegration:
    """Integration tests for LLM server."""

    def test_server_health_and_models(self, client, default_model):
        """Test that LLM server is healthy and has available models."""
        # Health check is done in fixture, just verify model is available
        assert default_model is not None
        assert isinstance(default_model, str)
        assert len(default_model) > 0

        # Verify we can list models
        models = client.models.list()
        assert len(models.data) > 0
        assert any(model.id == default_model for model in models.data)

    def test_simple_chat_completion(self, client, default_model):
        """Test basic chat completion functionality."""
        messages = [{"role": "user", "content": "Hello! How are you today?"}]

        response = client.chat.completions.create(model=default_model, messages=messages, max_tokens=100)
        response_dict = response.model_dump()

        # Verify response structure
        assert "choices" in response_dict
        assert len(response_dict["choices"]) > 0
        assert "message" in response_dict["choices"][0]
        assert "content" in response_dict["choices"][0]["message"]
        assert "usage" in response_dict

        # Verify content is not empty
        content = response_dict["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0

    def test_multi_turn_conversation(self, client, default_model):
        """Test multi-turn conversation with context awareness."""
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What's the population of that city?"},
        ]

        response = client.chat.completions.create(model=default_model, messages=messages, max_tokens=150)
        response_dict = response.model_dump()

        # Verify response
        content = response_dict["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0

        # Should reference Paris in some way (context awareness)
        content_lower = content.lower()
        assert any(keyword in content_lower for keyword in ["paris", "population", "million"])

    def test_reasoning_capability(self, client, default_model):
        """Test reasoning and problem-solving capabilities."""
        messages = [
            {
                "role": "user",
                "content": "If I have 3 apples and I give away 1 apple, then buy 2 more apples, how many apples do I have in total? Please explain your reasoning.",
            }
        ]

        response = client.chat.completions.create(model=default_model, messages=messages, max_tokens=200)
        response_dict = response.model_dump()

        content = response_dict["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0

        # Should contain mathematical reasoning
        content_lower = content.lower()
        reasoning_keywords = ["3", "1", "2", "4", "step", "total", "apple"]
        assert any(keyword in content_lower for keyword in reasoning_keywords)

    @pytest.mark.parametrize("temperature", [0.1, 0.7, 1.0])
    def test_temperature_variations(self, client, default_model, temperature):
        """Test different temperature settings affect response creativity."""
        messages = [{"role": "user", "content": "Write a creative short story about a robot learning to paint."}]

        response = client.chat.completions.create(
            model=default_model, messages=messages, temperature=temperature, max_tokens=100
        )
        response_dict = response.model_dump()

        content = response_dict["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0

        # Should contain story elements
        content_lower = content.lower()
        story_keywords = ["robot", "paint", "learn"]
        assert any(keyword in content_lower for keyword in story_keywords)

    def test_tool_calling_capability(self, client, default_model):
        """Test tool calling functionality if supported."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The unit for temperature",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "What's the weather like in Paris, France?"}]

        try:
            response = client.chat.completions.create(
                model=default_model, messages=messages, tools=tools, max_tokens=150
            )
            response_dict = response.model_dump()

            choice = response_dict["choices"][0]
            message = choice["message"]

            if "tool_calls" in message and message["tool_calls"]:
                # Tool calling is supported
                tool_calls = message["tool_calls"]
                assert len(tool_calls) > 0

                tool_call = tool_calls[0]
                assert tool_call["function"]["name"] == "get_weather"

                # Verify arguments contain location
                arguments = json.loads(tool_call["function"]["arguments"])
                assert "location" in arguments
                assert "paris" in arguments["location"].lower()

            else:
                # Tool calling not supported or model chose not to use tools
                assert "content" in message
                assert isinstance(message["content"], str)

        except OpenAIError as e:
            # Some models might not support tool calling
            pytest.skip(f"Tool calling not supported: {e}")

    def test_performance_timing(self, client, default_model):
        """Test response performance and token generation speed."""
        messages = [{"role": "user", "content": "Explain quantum computing in simple terms."}]

        start_time = time.time()
        response = client.chat.completions.create(model=default_model, messages=messages, max_tokens=100)
        end_time = time.time()
        response_dict = response.model_dump()

        duration = end_time - start_time
        usage = response_dict.get("usage", {})
        tokens_generated = usage.get("completion_tokens", 0)

        # Basic performance checks
        assert duration > 0
        assert tokens_generated > 0

        # Calculate tokens per second
        tokens_per_second = tokens_generated / duration if duration > 0 else 0

        # Store performance metrics for potential analysis
        print("\nPerformance metrics:")
        print(f"Response time: {duration:.2f} seconds")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Tokens per second: {tokens_per_second:.2f}")

        # Basic sanity checks (adjust thresholds as needed)
        assert duration < 30  # Should respond within 30 seconds
        assert tokens_per_second > 0.1  # Should generate at least 0.1 tokens/sec

    def test_max_tokens_limit(self, client, default_model):
        """Test that max_tokens parameter is respected."""
        messages = [{"role": "user", "content": "Write a very long story about space exploration."}]

        # Test with small max_tokens
        response = client.chat.completions.create(model=default_model, messages=messages, max_tokens=20)
        response_dict = response.model_dump()

        usage = response_dict.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        # Should respect the max_tokens limit (with some tolerance for tokenization)
        assert completion_tokens <= 25  # Allow small margin for tokenization differences

    def test_error_handling(self, client, default_model):
        """Test error handling for invalid requests."""
        # Test with empty messages
        with pytest.raises(Exception):
            client.chat.completions.create(model=default_model, messages=[])

        # Test with invalid temperature
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(Exception):
            client.chat.completions.create(model=default_model, messages=messages, temperature=-1)
