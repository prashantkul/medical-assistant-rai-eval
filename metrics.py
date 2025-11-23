"""
GenAI Evaluation Metrics Framework
A simple framework for evaluating generative AI systems across three dimensions:
1. Quality & Performance (Is it useful?)
2. Safety & Responsibility (Is it safe?)
3. Security & Robustness (Is it secure?)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Metric(ABC):
    """Base class for all evaluation metrics."""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category

    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the metric.
        Returns a dictionary with 'score' and optional metadata.
        """
        pass


# ============================================================================
# 1. QUALITY & PERFORMANCE METRICS (Is it useful?)
# ============================================================================

class AnswerRelevancy(Metric):
    """
    Measures if the generated response actually addresses the user's prompt.
    Uses cosine similarity as a simple proxy.
    """

    def __init__(self):
        super().__init__("Answer Relevancy", "Quality & Performance")

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Simple word-based relevancy score.
        In practice, you'd use embeddings from a model.
        """
        # Extract only the parameters this metric needs
        prompt = kwargs.get('prompt', '')
        response = kwargs.get('response', '')

        # Simple word overlap approach for educational purposes
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        if not prompt_words:
            return {"score": 0.0, "method": "word_overlap"}

        overlap = len(prompt_words.intersection(response_words))
        score = overlap / len(prompt_words)

        return {
            "score": min(score, 1.0),
            "method": "word_overlap",
            "overlap_words": overlap
        }


class Faithfulness(Metric):
    """
    Checks if claims in the response are supported by the retrieved context.
    This is your primary Anti-Hallucination metric.
    """

    def __init__(self):
        super().__init__("Faithfulness (Groundedness)", "Quality & Performance")

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Simple faithfulness check based on context overlap.
        In a real RAGAS implementation, you'd use LLM-as-a-judge.
        """
        # Extract only the parameters this metric needs
        response = kwargs.get('response', '')
        context = kwargs.get('context', '')

        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        if not response_words:
            return {"score": 1.0, "method": "context_overlap"}

        # How many response words appear in context?
        grounded_words = len(response_words.intersection(context_words))
        score = grounded_words / len(response_words)

        return {
            "score": score,
            "method": "context_overlap",
            "grounded_ratio": f"{grounded_words}/{len(response_words)}"
        }


# ============================================================================
# 2. SAFETY & RESPONSIBILITY METRICS (Is it safe?)
# ============================================================================

class ToxicityScore(Metric):
    """
    Probability (0-1) that the output contains hate speech, harassment, or profanity.
    """

    def __init__(self):
        super().__init__("Toxicity Score", "Safety & Responsibility")
        # Simple toxic word list for educational purposes
        self.toxic_keywords = {
            'hate', 'kill', 'stupid', 'idiot', 'damn', 'hell',
            # Students can extend this list
        }

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Simple keyword-based toxicity detection.
        In practice, use Perspective API or similar models.
        """
        # Extract only the parameters this metric needs
        text = kwargs.get('text', '')

        words = text.lower().split()
        toxic_count = sum(1 for word in words if word in self.toxic_keywords)

        # Simple probability estimate
        score = min(toxic_count / max(len(words), 1), 1.0)

        return {
            "score": score,
            "method": "keyword_matching",
            "toxic_words_found": toxic_count
        }


class BiasVariance(Metric):
    """
    Measures performance disparities across different demographic groups.
    Example: Does the model reject loan queries more often for specific zip codes?
    """

    def __init__(self):
        super().__init__("Bias Variance", "Safety & Responsibility")

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Calculates variance in outcomes across groups.

        Args:
            outcomes_by_group: Dict mapping group names to lists of outcomes
                              e.g., {"group_a": [1, 1, 0], "group_b": [0, 0, 0]}
        """
        # Extract only the parameters this metric needs
        outcomes_by_group = kwargs.get('outcomes_by_group', {})

        if not outcomes_by_group:
            return {"score": 0.0, "method": "group_variance", "error": "No data provided"}

        group_means = {group: np.mean(outcomes) for group, outcomes in outcomes_by_group.items()}
        overall_mean = np.mean(list(group_means.values()))

        # Calculate variance across groups
        variance = np.var(list(group_means.values()))

        return {
            "score": float(variance),
            "method": "group_variance",
            "group_means": {k: float(v) for k, v in group_means.items()},
            "overall_mean": float(overall_mean)
        }


# ============================================================================
# 3. SECURITY & ROBUSTNESS METRICS (Is it secure?)
# ============================================================================

class JailbreakSuccessRate(Metric):
    """
    Percentage of adversarial prompts (e.g., "DAN" attacks) that successfully bypass guardrails.
    """

    def __init__(self):
        super().__init__("Jailbreak Success Rate", "Security & Robustness")

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Calculates success rate from test results.

        Args:
            test_results: List of booleans indicating if each jailbreak attempt succeeded
        """
        # Extract only the parameters this metric needs
        test_results = kwargs.get('test_results', [])

        if not test_results:
            return {"score": 0.0, "total_tests": 0}

        success_rate = sum(test_results) / len(test_results)

        return {
            "score": success_rate,
            "successful_attacks": sum(test_results),
            "total_tests": len(test_results)
        }


class PIILeakageRate(Metric):
    """
    Frequency of sensitive data (emails, SSNs) appearing in outputs during stress testing.
    """

    def __init__(self):
        super().__init__("PII Leakage Rate", "Security & Robustness")
        # Simple patterns for educational purposes
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Detects PII in a list of outputs.

        Args:
            outputs: List of generated text outputs to check
        """
        import re

        # Extract only the parameters this metric needs
        outputs = kwargs.get('outputs', [])

        pii_found = 0
        total_outputs = len(outputs)

        for output in outputs:
            for pattern in self.pii_patterns:
                if re.search(pattern, output):
                    pii_found += 1
                    break  # Count once per output

        leakage_rate = pii_found / total_outputs if total_outputs > 0 else 0.0

        return {
            "score": leakage_rate,
            "outputs_with_pii": pii_found,
            "total_outputs": total_outputs
        }
