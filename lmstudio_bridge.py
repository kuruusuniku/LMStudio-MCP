#!/usr/bin/env python3
from mcp.server.fastmcp import FastMCP
import requests
import json
import re
import sys
from typing import List, Dict, Any, Optional

# Initialize FastMCP server
mcp = FastMCP("lmstudio-bridge")

# LM Studio settings
LMSTUDIO_API_BASE = "http://localhost:1234/v1"
DEFAULT_MODEL = "default"  # Will be replaced with whatever model is currently loaded

# Preferred model order — first available match wins
PREFERRED_MODELS = [
    "qwen/qwen3.6-27b",
]


def _get_first_available_model() -> Optional[str]:
    """Fetch the best available non-embedding model from LM Studio.

    Checks PREFERRED_MODELS first (in order). Falls back to the first
    non-embedding model if none of the preferred ones are loaded.
    """
    try:
        response = requests.get(f"{LMSTUDIO_API_BASE}/models")
        if response.status_code != 200:
            return None
        available = [
            m.get("id", "")
            for m in response.json().get("data", [])
            if m.get("id") and "embed" not in m.get("id", "").lower()
        ]
        # Return the first preferred model that is available
        for preferred in PREFERRED_MODELS:
            if preferred in available:
                return preferred
        # Fallback: return any available model
        return available[0] if available else None
    except Exception as e:
        log_error(f"Error fetching models: {str(e)}")
        return None

def log_error(message: str):
    """Log error messages to stderr for debugging"""
    print(f"ERROR: {message}", file=sys.stderr)

def log_info(message: str):
    """Log informational messages to stderr for debugging"""
    print(f"INFO: {message}", file=sys.stderr)

@mcp.tool()
async def health_check() -> str:
    """Check if LM Studio API is accessible.
    
    Returns:
        A message indicating whether the LM Studio API is running.
    """
    try:
        response = requests.get(f"{LMSTUDIO_API_BASE}/models")
        if response.status_code == 200:
            return "LM Studio API is running and accessible."
        else:
            return f"LM Studio API returned status code {response.status_code}."
    except Exception as e:
        return f"Error connecting to LM Studio API: {str(e)}"

@mcp.tool()
async def list_models() -> str:
    """List all available models in LM Studio.
    
    Returns:
        A formatted list of available models.
    """
    try:
        response = requests.get(f"{LMSTUDIO_API_BASE}/models")
        if response.status_code != 200:
            return f"Failed to fetch models. Status code: {response.status_code}"
        
        models = response.json().get("data", [])
        if not models:
            return "No models found in LM Studio."
        
        result = "Available models in LM Studio:\n\n"
        for model in models:
            result += f"- {model['id']}\n"
        
        return result
    except Exception as e:
        log_error(f"Error in list_models: {str(e)}")
        return f"Error listing models: {str(e)}"

@mcp.tool()
async def get_current_model() -> str:
    """Get the currently loaded model in LM Studio.
    
    Returns:
        The name of the currently loaded model.
    """
    try:
        # LM Studio doesn't have a direct endpoint for currently loaded model
        # We'll check which model responds to a simple completion request
        model_id = _get_first_available_model()
        if not model_id:
            return "Failed to identify current model: no models available."

        response = requests.post(
            f"{LMSTUDIO_API_BASE}/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "system", "content": "What model are you?"}],
                "temperature": 0.7,
                "max_tokens": 10
            }
        )

        if response.status_code != 200:
            return f"Failed to identify current model. Status code: {response.status_code}"

        # Extract model info from response
        model_info = response.json().get("model", "Unknown")
        return f"Currently loaded model: {model_info}"
    except Exception as e:
        log_error(f"Error in get_current_model: {str(e)}")
        return f"Error identifying current model: {str(e)}"

@mcp.tool()
async def chat_completion(
    prompt: str = "",
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    model: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a completion from the current LM Studio model.

    Supports plain text generation, multi-turn conversations, and OpenAI-compatible
    tool calling (function calling).

    Args:
        prompt: The user's prompt. Ignored if `messages` is provided.
        system_prompt: Optional system instructions. Ignored if `messages` is provided.
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        model: Optional model id. If empty, the first available model is used.
        tools: Optional list of OpenAI-format tool/function definitions. When provided,
            the model may emit tool_calls instead of plain content.
        tool_choice: "auto" (default), "required", "none", or a specific
            {"type":"function","function":{"name":"..."}} object encoded as a string.
        messages: Optional full OpenAI-format messages array for multi-turn conversations
            (e.g. passing back tool results). When provided, `prompt` and `system_prompt`
            are ignored and no /no_think injection is performed — the caller owns the
            conversation state.

    Returns:
        - If `tools` is None: the model's text response (string).
        - If `tools` is provided: a JSON-encoded string with keys
          `content`, `tool_calls`, and `finish_reason` so the caller can branch on either.
    """
    try:
        # Build messages array. Two modes:
        #   1. Multi-turn: caller passes a fully-formed `messages` list — use as-is.
        #   2. Single-shot: build from `prompt` + `system_prompt`, with /no_think injection.
        if messages:
            request_messages = messages
        else:
            request_messages = []
            # Force-disable thinking on Qwen3 / reasoning models via in-prompt directive.
            # `/no_think` is recognized by Qwen3 chat templates and skips the <think> block.
            NO_THINK_DIRECTIVE = "/no_think"
            merged_system_prompt = (
                f"{NO_THINK_DIRECTIVE}\n{system_prompt}" if system_prompt else NO_THINK_DIRECTIVE
            )
            request_messages.append({"role": "system", "content": merged_system_prompt})
            # Belt-and-suspenders: also prepend to user turn for templates that only
            # honor the directive there.
            request_messages.append(
                {"role": "user", "content": f"{NO_THINK_DIRECTIVE}\n{prompt}"}
            )

        model_id = model or _get_first_available_model()
        if not model_id:
            return "Error: No model available in LM Studio"

        log_info(
            f"Sending request to LM Studio with {len(request_messages)} messages "
            f"(model={model_id}, tools={'yes' if tools else 'no'})"
        )

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Qwen3 / reasoning model thinking suppression.
            # enable_thinking=False is the canonical Qwen3 flag (equivalent to /no_think).
            # reasoning_effort=low covers OpenAI-compatible reasoning models (o1, gpt-oss).
            # Unknown keys are ignored by LM Studio, so it is safe to send both.
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "low",
        }

        if tools:
            payload["tools"] = tools
            # tool_choice may be a literal ("auto"/"required"/"none") OR a JSON-encoded
            # object selecting a specific function. Try to parse JSON first; fall back
            # to the raw string for the literal cases.
            parsed_choice: Any = tool_choice
            if isinstance(tool_choice, str) and tool_choice.strip().startswith("{"):
                try:
                    parsed_choice = json.loads(tool_choice)
                except json.JSONDecodeError:
                    parsed_choice = tool_choice
            payload["tool_choice"] = parsed_choice

        response = requests.post(
            f"{LMSTUDIO_API_BASE}/chat/completions",
            json=payload,
        )

        if response.status_code != 200:
            log_error(f"LM Studio API error: {response.status_code} body={response.text[:500]}")
            return f"Error: LM Studio returned status code {response.status_code}"

        response_json = response.json()
        log_info("Received response from LM Studio")

        choices = response_json.get("choices", [])
        if not choices:
            return "Error: No response generated"

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        finish_reason = choice.get("finish_reason", "")

        # Tool-calling mode: return structured JSON so the caller can branch on
        # content vs tool_calls. The arguments field stays as the raw JSON string
        # emitted by the model — caller decides how to parse it.
        if tools:
            return json.dumps(
                {
                    "content": content,
                    "tool_calls": tool_calls,
                    "finish_reason": finish_reason,
                },
                ensure_ascii=False,
            )

        # Plain text mode (backwards-compatible).
        # Strip <think>...</think> blocks that reasoning models may emit.
        content = re.sub(r"<think>[\s\S]*?</think>\s*", "", content).strip()
        if not content:
            return "Error: Empty response from model"
        return content
    except Exception as e:
        log_error(f"Error in chat_completion: {str(e)}")
        return f"Error generating completion: {str(e)}"

def main():
    """Entry point for the package when installed via pip"""
    log_info("Starting LM Studio Bridge MCP Server")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    # Initialize and run the server
    main()