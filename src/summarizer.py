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

        prompt = f"""You are an expert at summarizing meeting transcripts. Analyze the following meeting transcription and create a comprehensive markdown summary.

{title_section}Meeting Transcription:
{transcription}

Please provide a detailed markdown summary with the following structure:

# Meeting Summary

## Key Points
- List the 5-7 most important decisions, action items, or key points discussed

## Action Items
- [ ] Specific task 1 (assignee if mentioned)
- [ ] Specific task 2 (assignee if mentioned)
- [ ] etc.

## Discussion Topics
Brief overview of main topics covered

## Decisions Made
- Decision 1
- Decision 2
- etc.

## Next Steps
What are the follow-up actions or next meeting topics?

## Attendees (if mentioned)
- Names and roles if mentioned in the transcription

Focus on clarity, actionability, and capturing the essential information. Use proper markdown formatting with headers, bullet points, and checkboxes for action items."""

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
