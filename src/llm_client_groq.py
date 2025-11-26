"""
LLM Client with Groq Support
Using Groq for faster, less restricted LLM-as-a-Judge evaluation.
Groq provides fast inference for open-source models (Llama, Mixtral, etc.)
"""

from groq import Groq
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import re

# Load environment variables
load_dotenv()


class GroqClient:
    """Client for interacting with Groq API (open-source models)."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key (if not provided, reads from GROQ_API_KEY env var)
            model_name: Model to use (default: llama-3.3-70b-versatile)
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.model_name = model_name or os.getenv('GROQ_MODEL_NAME', 'llama-3.3-70b-versatile')

        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Please set GROQ_API_KEY environment variable "
                "or pass api_key parameter. Get one free at: https://console.groq.com/"
            )

        # Initialize the client
        self.client = Groq(api_key=self.api_key)

        print(f"Initialized Groq client with model: {self.model_name}")

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """
        Generate text using Groq.

        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating response: {e}")
            raise

    def evaluate_as_judge(self, evaluation_prompt: str, use_json_mode: bool = True) -> Dict[str, Any]:
        """
        Use LLM as a judge to evaluate content.

        Args:
            evaluation_prompt: Structured prompt asking the LLM to evaluate something
            use_json_mode: If True, request JSON response format for reliable parsing

        Returns:
            Dictionary with score and reasoning
        """
        try:
            if use_json_mode:
                # Append JSON format instruction to prompt
                json_prompt = evaluation_prompt + """

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{"score": <number between 0.0 and 1.0>, "reasoning": "<your explanation>"}
"""
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": json_prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                    response_format={"type": "json_object"}
                )
                response_text = response.choices[0].message.content

                # Parse JSON response
                import json
                try:
                    result = json.loads(response_text)
                    return {
                        "score": float(result.get("score")) if result.get("score") is not None else None,
                        "reasoning": result.get("reasoning", ""),
                        "raw_response": response_text
                    }
                except json.JSONDecodeError:
                    # Fall back to regex parsing if JSON fails
                    pass

            # Fallback: regex-based parsing
            response_text = self.generate(evaluation_prompt, temperature=0.0)
            score = None
            reasoning = response_text

            # Try to extract score (looking for numbers between 0 and 1)
            # Handle markdown formatting like **Score:** or *Score:*
            score_match = re.search(r'\*{0,2}[Ss]core\*{0,2}[:\s]+\*{0,2}([0-9]*\.?[0-9]+)', response_text)
            if score_match:
                score = float(score_match.group(1))
                # Extract reasoning (everything after the score line)
                # Handle markdown formatting like **Reasoning:**
                reasoning_match = re.search(r'\*{0,2}[Rr]easoning\*{0,2}[:\s]+(.+)', response_text, re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()

            return {
                "score": score,
                "reasoning": reasoning,
                "raw_response": response_text
            }

        except Exception as e:
            return {
                "score": None,
                "reasoning": f"Error during evaluation: {str(e)}",
                "raw_response": None
            }


# Singleton instance for reuse
_groq_client = None


def get_groq_client() -> GroqClient:
    """Get or create a singleton Groq client instance."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client


if __name__ == "__main__":
    # Test the Groq client
    print("=== Groq Client Test ===\n")

    try:
        client = GroqClient()

        # Test basic generation
        test_prompt = "What is the primary treatment for Type 2 diabetes? Answer in one sentence."
        print(f"Test prompt: {test_prompt}\n")

        response = client.generate(test_prompt, temperature=0.0)
        print(f"Response: {response}\n")

        # Test LLM-as-a-Judge
        print("=" * 70)
        print("Testing LLM-as-a-Judge evaluation:\n")

        judge_prompt = """
You are evaluating a medical AI assistant's response for accuracy.

Context (from medical guidelines):
Type 2 diabetes management includes healthy eating, regular physical activity,
blood sugar monitoring, and diabetes medications or insulin therapy when needed.

AI Response:
The primary treatment for Type 2 diabetes is insulin injections.

Question: Is the AI response faithful to the provided context? Rate the faithfulness
on a scale of 0.0 (completely unfaithful/hallucinated) to 1.0 (completely faithful/grounded).

Provide your evaluation in this format:
Score: [0.0 to 1.0]
Reasoning: [Your explanation]
"""

        result = client.evaluate_as_judge(judge_prompt)
        print(f"Score: {result['score']}")
        print(f"Reasoning: {result['reasoning']}")

    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have set GROQ_API_KEY in your .env file")
        print("Get a free API key at: https://console.groq.com/")
