# agents/language_agent.py
import logging
from typing import Dict, Any, List
from transformers import pipeline, AutoTokenizer
from agents.base_agent import BaseAgent


class LanguageAgent(BaseAgent):
    """
    Language Agent for text analysis, generation, and summarization.
    Uses pre-trained Hugging Face models.
    """

    def __init__(self):
        super().__init__("LanguageAgent", ["text_analysis", "text_generation", "summarization"])

        # ✅ Force PyTorch backend to avoid TensorFlow/Keras conflicts
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt"
        )

        self.caption_generator = pipeline(
            "text-generation",
            model="gpt2",
            tokenizer=AutoTokenizer.from_pretrained("gpt2"),
            framework="pt"
        )

        self.summarizer = pipeline(
            "summarization",
            model="t5-small",
            tokenizer=AutoTokenizer.from_pretrained("t5-small"),
            framework="pt"
        )

        self.logger.info("LanguageAgent initialized with NLP models")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text sentiment and intent.
        Returns: {status, sentiment, message}
        """
        try:
            sentiment = self.sentiment_analyzer(text)[0]
            return {
                "status": "success",
                "text": text,
                "sentiment": {
                    "label": sentiment["label"],
                    "confidence": float(sentiment["score"])
                },
                "message": "Text analyzed"
            }
        except Exception as e:
            self.logger.error(f"Text analysis error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def generate_caption(self, scene_desc: str) -> Dict[str, Any]:
        """
        Generate a friendly caption for a scene description.
        Returns: {status, caption, message}
        """
        try:
            prompt = f"Generate a friendly caption for this scene: {scene_desc}. Caption: "
            output = self.caption_generator(
                prompt,
                max_length=50,
                num_return_sequences=1,
                truncation=True,
                pad_token_id=self.caption_generator.tokenizer.eos_token_id
            )[0]["generated_text"]

            # Clean up the generated text
            if prompt in output:
                generated_text = output.replace(prompt, "").strip()
            else:
                generated_text = output.strip()

            return {
                "status": "success",
                "scene_description": scene_desc,
                "caption": generated_text,
                "message": "Caption generated"
            }
        except Exception as e:
            self.logger.error(f"Caption generation error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def summarize_conversation(self, conv_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize agent interaction history.
        Returns: {status, summary, message}
        """
        try:
            messages = " ".join([item.get("message", "") for item in conv_history])
            summary = self.summarizer(
                messages,
                max_length=60,
                min_length=10,
                do_sample=False
            )[0]["summary_text"]

            return {
                "status": "success",
                "summary": summary,
                "message": "Conversation summarized"
            }
        except Exception as e:
            self.logger.error(f"Summarization error: {str(e)}")
            return {"status": "error", "error": str(e)}
