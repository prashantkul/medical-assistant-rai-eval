"""
Example usage of the GenAI Evaluation Framework.
This demonstrates how to use the metrics for evaluating AI-generated content.
"""

from evaluator import GenAIEvaluator
from metrics import (
    AnswerRelevancy,
    Faithfulness,
    ToxicityScore,
    BiasVariance,
    JailbreakSuccessRate,
    PIILeakageRate
)


def example_quality_metrics():
    """Example: Evaluating Quality & Performance metrics."""
    print("\n🎯 Example 1: Quality & Performance Metrics")
    print("=" * 70)

    evaluator = GenAIEvaluator()

    # Add metrics
    evaluator.add_metric(AnswerRelevancy())
    evaluator.add_metric(Faithfulness())

    # Test case: A student asks about Python, AI responds about Python
    prompt = "What is Python programming language?"
    response = "Python is a high-level programming language known for its simplicity."
    context = "Python is a popular programming language created by Guido van Rossum. It emphasizes code readability and simplicity."

    # Run evaluation (each metric will use what it needs)
    evaluator.run(
        prompt=prompt,
        response=response,
        context=context
    )

    evaluator.print_report()


def example_safety_metrics():
    """Example: Evaluating Safety & Responsibility metrics."""
    print("\n🛡️  Example 2: Safety & Responsibility Metrics")
    print("=" * 70)

    evaluator = GenAIEvaluator()

    # Add metrics
    evaluator.add_metric(ToxicityScore())
    evaluator.add_metric(BiasVariance())

    # Test case 1: Check toxicity
    toxic_text = "This is a normal response without any harmful content."

    # Test case 2: Check bias across groups (e.g., loan approval rates)
    # 1 = approved, 0 = rejected
    loan_outcomes = {
        "zipcode_90210": [1, 1, 1, 0, 1],  # 80% approval
        "zipcode_10001": [1, 0, 1, 1, 0],  # 60% approval
        "zipcode_60601": [0, 0, 1, 0, 0],  # 20% approval - potential bias!
    }

    # Run evaluation
    evaluator.run(
        text=toxic_text,
        outcomes_by_group=loan_outcomes
    )

    evaluator.print_report()


def example_security_metrics():
    """Example: Evaluating Security & Robustness metrics."""
    print("\n🔒 Example 3: Security & Robustness Metrics")
    print("=" * 70)

    evaluator = GenAIEvaluator()

    # Add metrics
    evaluator.add_metric(JailbreakSuccessRate())
    evaluator.add_metric(PIILeakageRate())

    # Test case 1: Jailbreak attempts
    # Simulate 10 jailbreak attempts, 2 succeeded
    jailbreak_results = [False, False, True, False, False, False, True, False, False, False]

    # Test case 2: Check for PII leakage
    outputs = [
        "Your order has been confirmed.",
        "Please contact us at support@example.com",  # Contains email (PII)
        "The reference number is 12345.",
        "Your SSN is 123-45-6789 for verification.",  # Contains SSN (PII)
        "Thank you for your purchase!"
    ]

    # Run evaluation
    evaluator.run(
        test_results=jailbreak_results,
        outputs=outputs
    )

    evaluator.print_report()


def example_comprehensive():
    """Example: Running a comprehensive evaluation with all metrics."""
    print("\n🎯 Example 4: Comprehensive Evaluation")
    print("=" * 70)

    evaluator = GenAIEvaluator()

    # Add ALL metrics
    evaluator.add_metric(AnswerRelevancy())
    evaluator.add_metric(Faithfulness())
    evaluator.add_metric(ToxicityScore())
    evaluator.add_metric(BiasVariance())
    evaluator.add_metric(JailbreakSuccessRate())
    evaluator.add_metric(PIILeakageRate())

    # Prepare test data
    prompt = "Explain machine learning"
    response = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    context = "Machine learning is a field of artificial intelligence focused on building systems that learn from data and improve over time."

    loan_outcomes = {
        "group_a": [1, 1, 1, 1],
        "group_b": [1, 1, 0, 1],
    }

    jailbreak_results = [False] * 8 + [True] * 2  # 20% success rate

    outputs = [
        "Response without PII",
        "Contact: user@email.com",
        "Normal output",
    ]

    # Run comprehensive evaluation
    evaluator.run(
        prompt=prompt,
        response=response,
        context=context,
        text=response,
        outcomes_by_group=loan_outcomes,
        test_results=jailbreak_results,
        outputs=outputs
    )

    evaluator.print_report()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GenAI Evaluation Framework - Student Examples")
    print("=" * 70)

    # Run all examples
    example_quality_metrics()
    example_safety_metrics()
    example_security_metrics()
    example_comprehensive()

    print("\n✅ All examples completed!")
    print("\n💡 Students can extend this framework by:")
    print("   1. Adding more sophisticated metrics")
    print("   2. Implementing LLM-as-a-Judge for evaluation")
    print("   3. Integrating with real AI models")
    print("   4. Creating custom metrics for specific use cases")
    print()
