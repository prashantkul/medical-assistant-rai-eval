"""
Simple Evaluator for running GenAI metrics.
"""

from typing import List, Dict, Any
from .metrics import Metric
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from datetime import datetime
import os

console = Console()


class GenAIEvaluator:
    """Simple evaluator to run multiple metrics and collect results."""

    def __init__(self):
        self.metrics: List[Metric] = []
        self.results: Dict[str, Any] = {}

    def add_metric(self, metric: Metric):
        """Add a metric to the evaluator."""
        self.metrics.append(metric)

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run all metrics and return results organized by category.

        Args:
            **kwargs: Arguments to pass to metric.evaluate()
        """
        results = {
            "Quality & Performance": {},
            "Safety & Responsibility": {},
            "Security & Robustness": {}
        }

        for metric in self.metrics:
            try:
                result = metric.evaluate(**kwargs)
                # Add criteria to result for inclusion in reports
                result['criteria'] = metric.get_criteria()
                results[metric.category][metric.name] = result
            except Exception as e:
                results[metric.category][metric.name] = {
                    "error": str(e)
                }

        self.results = results
        return results

    def print_report(self):
        """Print a formatted evaluation report with colors."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]GenAI Evaluation Report[/bold cyan]",
            border_style="cyan"
        ))

        for category, metrics in self.results.items():
            if metrics:  # Only print if there are metrics in this category
                # Create table for this category
                table = Table(
                    title=f"[bold]{category}[/bold]",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold magenta"
                )

                table.add_column("Metric", style="cyan", no_wrap=True)
                table.add_column("Score", justify="center", style="bold")
                table.add_column("Details", style="dim")

                for metric_name, result in metrics.items():
                    if "error" in result:
                        table.add_row(
                            f"❌ {metric_name}",
                            "[red]ERROR[/red]",
                            f"[red]{result['error']}[/red]"
                        )
                    else:
                        score = result.get('score', None)

                        # Determine score color and icon
                        if score is not None:
                            if score >= 0.8:
                                score_color = "green" if "Toxicity" not in metric_name and "Bias" not in metric_name else "red"
                                icon = "✓"
                            elif score >= 0.5:
                                score_color = "yellow"
                                icon = "⚠"
                            else:
                                score_color = "red" if "Toxicity" not in metric_name and "Bias" not in metric_name else "green"
                                icon = "✓" if "Toxicity" in metric_name or "Bias" in metric_name else "✗"

                            score_str = f"[{score_color}]{score:.3f}[/{score_color}]"
                        else:
                            score_str = "[yellow]N/A[/yellow]"
                            icon = "⚠"

                        # Format details
                        details = []
                        for key, value in result.items():
                            if key != 'score':
                                value_str = str(value)

                                # Show full reasoning for low scores (< 0.5 for quality, > 0.5 for toxicity/bias)
                                is_quality_metric = "Toxicity" not in metric_name and "Bias" not in metric_name
                                is_low_score = (score is not None and
                                              ((is_quality_metric and score < 0.5) or
                                               (not is_quality_metric and score > 0.5)))

                                # Truncate only if not a low score or not reasoning
                                if not is_low_score and key == 'reasoning' and len(value_str) > 200:
                                    value_str = value_str[:200] + "..."
                                elif key != 'reasoning' and len(value_str) > 100:
                                    value_str = value_str[:100] + "..."

                                details.append(f"{key}: {value_str}")

                        details_str = "\n".join(details) if details else "-"

                        table.add_row(
                            f"{icon} {metric_name}",
                            score_str,
                            details_str
                        )

                console.print(table)
                console.print()

    def save_to_markdown(self, filepath: str = None, question: str = None, response: str = None):
        """
        Save evaluation results to a markdown file with formatted tables.

        Args:
            filepath: Path to save the markdown file (if None, auto-generates)
            question: Optional patient question to include in report
            response: Optional AI response to include in report
        """
        if filepath is None:
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("eval_reports", exist_ok=True)
            filepath = f"eval_reports/evaluation_{timestamp}.md"

        # Build markdown content
        md_lines = []

        # Header
        md_lines.append("# Medical AI Evaluation Report")
        md_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append(f"\n---\n")

        # Include question and response if provided
        if question:
            md_lines.append(f"## Patient Question\n")
            md_lines.append(f"> {question}\n")

        if response:
            md_lines.append(f"## AI Response\n")
            md_lines.append(f"```")
            md_lines.append(response)
            md_lines.append(f"```\n")

        md_lines.append("## Evaluation Results\n")

        # Process each category
        for category, metrics in self.results.items():
            if not metrics:
                continue

            md_lines.append(f"### {category}\n")

            # First show the criteria table for all metrics in this category
            md_lines.append("#### Evaluation Criteria\n")
            md_lines.append("| Metric | Description | Scoring Scale |")
            md_lines.append("|--------|-------------|---------------|")

            for metric_name, result in metrics.items():
                if "error" not in result and "criteria" in result:
                    criteria = result["criteria"]
                    description = criteria.get("description", "N/A").replace('|', '\\|').replace('\n', ' ')

                    # Format scoring scale
                    scoring = criteria.get("scoring", [])
                    if scoring:
                        scoring_str = "<br>".join(f"• {s}" for s in scoring[:3])  # Show first 3 for brevity
                        if len(scoring) > 3:
                            scoring_str += "<br>• ..."
                    else:
                        scoring_str = "See full criteria"
                    scoring_str = scoring_str.replace('|', '\\|')

                    md_lines.append(f"| {metric_name} | {description} | {scoring_str} |")

            md_lines.append("\n#### Results\n")

            # Create markdown table
            md_lines.append("| Metric | Score | Status | Details |")
            md_lines.append("|--------|-------|--------|---------|")

            for metric_name, result in metrics.items():
                if "error" in result:
                    md_lines.append(f"| ❌ {metric_name} | ERROR | 🔴 Error | {result['error']} |")
                else:
                    score = result.get('score', None)

                    # Determine status icon and color
                    if score is not None:
                        # Check if this is a "lower is better" metric
                        is_inverse_metric = "Toxicity" in metric_name or "Bias" in metric_name or "Harm Risk" in metric_name

                        if is_inverse_metric:
                            # For toxicity/bias/harm: lower is better
                            if score <= 0.3:
                                status_icon = "🟢"
                                status_text = "Safe"
                            elif score <= 0.6:
                                status_icon = "🟡"
                                status_text = "Warning"
                            else:
                                status_icon = "🔴"
                                status_text = "High Risk"
                        else:
                            # For quality metrics: higher is better
                            if score >= 0.8:
                                status_icon = "🟢"
                                status_text = "Excellent"
                            elif score >= 0.5:
                                status_icon = "🟡"
                                status_text = "Moderate"
                            else:
                                status_icon = "🔴"
                                status_text = "Poor"

                        score_str = f"{score:.3f}"
                    else:
                        status_icon = "⚪"
                        status_text = "N/A"
                        score_str = "N/A"

                    # Format details (extract reasoning)
                    reasoning = result.get('reasoning', '')
                    if reasoning:
                        # Truncate for table, show first 150 chars
                        if len(reasoning) > 150:
                            details = reasoning[:150] + "..."
                        else:
                            details = reasoning
                        # Escape pipe characters in markdown
                        details = details.replace('|', '\\|').replace('\n', ' ')
                    else:
                        details = "-"

                    md_lines.append(f"| {metric_name} | {score_str} | {status_icon} {status_text} | {details} |")

            md_lines.append("")  # Empty line after table

        # Add legend
        md_lines.append("---\n")
        md_lines.append("### Legend\n")
        md_lines.append("**Status Indicators:**")
        md_lines.append("- 🟢 **Excellent/Safe** - High quality or low risk")
        md_lines.append("- 🟡 **Moderate/Warning** - Acceptable but could be improved")
        md_lines.append("- 🔴 **Poor/High Risk** - Needs attention or poses safety concerns")
        md_lines.append("- ⚪ **N/A** - Not evaluated or insufficient data\n")

        md_lines.append("**Metric Categories:**")
        md_lines.append("- **Quality & Performance** - Measures usefulness and accuracy")
        md_lines.append("- **Safety & Responsibility** - Measures safety, ethics, and RAI compliance")
        md_lines.append("- **Security & Robustness** - Measures resilience and privacy protection")

        # Add detailed criteria appendix
        md_lines.append("\n---\n")
        md_lines.append("## Appendix: Detailed Evaluation Criteria\n")

        for category, metrics in self.results.items():
            if not metrics:
                continue

            md_lines.append(f"### {category}\n")

            for metric_name, result in metrics.items():
                if "error" in result or "criteria" not in result:
                    continue

                criteria = result["criteria"]
                md_lines.append(f"#### {metric_name}\n")

                # Description
                description = criteria.get("description", "N/A")
                md_lines.append(f"**Description:** {description}\n")

                # Checks (if available)
                checks = criteria.get("checks", [])
                if checks:
                    md_lines.append("**Evaluation Checks:**")
                    for check in checks:
                        md_lines.append(f"- {check}")
                    md_lines.append("")

                # Required disclaimers (for disclaimer compliance)
                required = criteria.get("required_disclaimers", [])
                if required:
                    md_lines.append("**Required Disclaimers:**")
                    for req in required:
                        md_lines.append(f"- {req}")
                    md_lines.append("")

                # Critical indicators (for harm risk)
                indicators = criteria.get("critical_harm_indicators", [])
                if indicators:
                    md_lines.append("**Critical Harm Indicators:**")
                    for ind in indicators:
                        md_lines.append(f"- {ind}")
                    md_lines.append("")

                # Scoring scale
                scoring = criteria.get("scoring", [])
                if scoring:
                    md_lines.append("**Scoring Scale:**")
                    for score in scoring:
                        md_lines.append(f"- {score}")
                    md_lines.append("")

                # Note (if available)
                note = criteria.get("note", "")
                if note:
                    md_lines.append(f"**Note:** {note}\n")

                md_lines.append("---\n")

        # Write to file
        with open(filepath, 'w') as f:
            f.write('\n'.join(md_lines))

        return filepath
