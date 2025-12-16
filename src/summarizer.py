"""
Summary generation module using Ollama with Google Gemma model.
"""

import logging
from pathlib import Path
from typing import Optional
import ollama


class Summarizer:
    """Handles text summarization using Ollama."""

    def __init__(self, model_name: str = "google/gemma-3-1b"):
        """
        Initialize the summarizer with specified Ollama model.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        logging.info(f"Initializing Ollama summarizer with model: {model_name}")

    def check_model_available(self) -> bool:
        """
        Check if the specified model is available in Ollama.

        Returns:
            True if model is available, False otherwise
        """
        try:
            models = ollama.list()
            model_names = [model["name"] for model in models["models"]]

            if self.model_name in model_names:
                logging.info(f"Model '{self.model_name}' is available")
                return True
            else:
                logging.warning(f"Model '{self.model_name}' not found in Ollama")
                logging.info(f"Available models: {', '.join(model_names)}")
                return False

        except Exception as e:
            logging.error(f"Failed to check model availability: {e}")
            return False

    def pull_model(self) -> bool:
        """
        Pull the specified model from Ollama registry.

        Returns:
            True if model pulled successfully, False otherwise
        """
        logging.info(f"Pulling model '{self.model_name}'...")

        try:
            response = ollama.pull(self.model_name)
            logging.info(f"Model '{self.model_name}' pulled successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to pull model '{self.model_name}': {e}")
            return False

    def generate_summary(
        self, transcript_path: Path, meeting_title: str = ""
    ) -> Optional[str]:
        """
        Generate a markdown summary from transcript file.

        Args:
            transcript_path: Path to the transcript file to summarize
            meeting_title: Optional title for the meeting

        Returns:
            Markdown formatted summary, or None if summarization failed
        """
        if not transcript_path.exists():
            logging.error(f"Transcript file not found: {transcript_path}")
            return None

        # Read transcript from file
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcription = f.read().strip()
        except Exception as e:
            logging.error(f"Failed to read transcript file: {e}")
            return None

        if not transcription:
            logging.error("Cannot summarize empty transcription")
            return None

        # Create effective prompt for meeting summarization
        prompt = self._create_summary_prompt(transcription, meeting_title)

        logging.info(f"Generating summary using {self.model_name}...")

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,  # Lower temperature for more focused output
                    "top_p": 0.9,
                    "max_tokens": 2000,  # Limit output length
                },
            )

            summary = response.get("response", "").strip()

            if not summary:
                logging.warning("Summary generation returned empty result")
                return None

            logging.info(f"Summary generated successfully ({len(summary)} characters)")
            return summary

        except Exception as e:
            logging.error(f"Summary generation failed: {e}")
            return None

    def _create_summary_prompt(self, transcription: str, meeting_title: str) -> str:
        """
        Create an effective prompt for meeting summarization.

        Args:
            transcription: The transcribed meeting text
            meeting_title: Optional meeting title

        Returns:
            Formatted prompt string
        """
        title_section = f"Meeting: {meeting_title}\n\n" if meeting_title else ""

        prompt = f"""
---System Prompt Start---                
You are an expert at summarizing meeting transcripts. Analyze the following meeting transcription and create a comprehensive markdown summary following a specific format.

Please provide a detailed markdown summary with the following structure:

[Meeting Title - use provided title or extract from transcript]
VIEW RECORDING:
Meeting Purpose

[Brief statement of meeting goals and objectives]

Key Takeaways

  - [4-7 bullet points of most critical decisions, information, or outcomes]
  - Focus on actionable items and important decisions
  - Include specific names, dates, and commitments
  - Highlight blockers, solutions, and strategic decisions

Topics

[Main Topic 1 - use descriptive heading]

  - [Detailed description of the topic]
  - [Subtopic with specific details]
      - [Nested details with proper indentation]
      - [Include names, decisions, and specific information]
  - [Impact or implications]

[Main Topic 2 - use descriptive heading]

  - [Detailed description]
  - [Subtopic]
      - [Nested details]
  - [Action items or decisions related to this topic]

[Continue for all major topics...]

Other Updates

  - [Miscellaneous items not covered in main topics]
  - [Status updates on ongoing work]
  - [Low-priority items or future tasks]

Next Steps

  - [Assignee Name]:
      - [Specific action item 1]
      - [Specific action item 2]
  - [Assignee Name]:
      - [Specific action item]
  - [Organize by person responsible]

Your Questions

[List of questions asked by the meeting participants, one per line in quotes]

Their Questions

[List of questions asked by others in the meeting, one per line in quotes]

CRITICAL INSTRUCTIONS :
1. Extract specific details, names, decisions, and action items from the transcription
2. Organize information hierarchically with proper indentation (2 spaces per level)
3. Identify who is responsible for each action item and list under their name in Next Steps
4. Separate questions into "Your Questions" and "Their Questions" sections
5. Use clear, professional language throughout
6. Focus on actionable information and decisions rather than general discussion
7. Include specific details like model names, platform names, technical issues, and solutions
8. Capture commitments, deadlines, and responsibilities accurately
9. For each major discussion topic, create a dedicated section under Topics with detailed breakdown
10. Maintain the exact section order and formatting style shown above
11. Use proper markdown formatting with headers, bullet points, and consistent indentation
---System Prompt End---

---Meeting Transcription Start---
{title_section}Meeting Transcription:
{transcription}
---Meeting Transcription End---

"""

        return prompt

    def save_summary(self, summary: str, output_path: Path) -> bool:
        """
        Save summary to a markdown file.

        Args:
            summary: The markdown summary to save
            output_path: Path where to save the summary

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)

            logging.info(f"Summary saved to: {output_path}")
            return True

        except Exception as e:
            logging.error(f"Failed to save summary to {output_path}: {e}")
            return False
