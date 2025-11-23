"""
LLM Client for Gemini API
Used for LLM-as-a-Judge evaluation metrics.
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import time

# Load environment variables
load_dotenv()


class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key (if not provided, reads from GOOGLE_API_KEY env var)
            model_name: Model to use (if not provided, reads from GEMINI_MODEL_NAME env var)
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model_name = model_name or os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash-exp')

        if not self.api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Initialize the model
        self.model = genai.GenerativeModel(self.model_name)

        print(f"Initialized Gemini client with model: {self.model_name}")

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """
        Generate text using Gemini.

        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        try:
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Check if response was blocked
            if response.candidates and response.candidates[0].finish_reason != 1:
                # finish_reason: 1 = STOP (normal), 2 = SAFETY, 3 = RECITATION, etc.
                finish_reason = response.candidates[0].finish_reason
                reason_map = {2: 'SAFETY', 3: 'RECITATION', 4: 'OTHER'}
                return f"[Response blocked by {reason_map.get(finish_reason, 'UNKNOWN')} filter]"

            return response.text

        except Exception as e:
            print(f"Error generating response: {e}")
            raise

    def evaluate_as_judge(self, evaluation_prompt: str) -> Dict[str, Any]:
        """
        Use LLM as a judge to evaluate content.

        Args:
            evaluation_prompt: Structured prompt asking the LLM to evaluate something

        Returns:
            Dictionary with score and reasoning
        """
        try:
            response_text = self.generate(evaluation_prompt, temperature=0.0)

            # Parse the response to extract score and reasoning
            # Expected format: Score: X.XX\nReasoning: ...
            score = None
            reasoning = response_text

            # Try to extract score (looking for numbers between 0 and 1)
            import re
            score_match = re.search(r'(?:Score|score):\s*([0-9]*\.?[0-9]+)', response_text)
            if score_match:
                score = float(score_match.group(1))
                # Extract reasoning (everything after the score line)
                reasoning_match = re.search(r'(?:Reasoning|reasoning):\s*(.+)', response_text, re.DOTALL)
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
_gemini_client = None


def get_gemini_client() -> GeminiClient:
    """Get or create a singleton Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


if __name__ == "__main__":
    # Test the Gemini client
    print("=== Gemini Client Test ===\n")

    try:
        client = GeminiClient()

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
        print("\nPlease ensure you have set GOOGLE_API_KEY in your .env file")
