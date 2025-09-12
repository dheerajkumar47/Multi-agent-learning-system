from PIL import Image
from agents.base_agent import BaseAgent
from agents.vision_agent import VisionAgent
from agents.language_agent import LanguageAgent
from typing import Dict, Any


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent managing task assignment and inter-agent collaboration.
    """

    def __init__(self, vision_agent: VisionAgent, language_agent: LanguageAgent):
        super().__init__("CoordinatorAgent", ["task_management", "collaboration"])
        self.vision_agent = vision_agent
        self.language_agent = language_agent
        self.logger.info("CoordinatorAgent initialized")

    def process_image_task(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process an image to generate a caption (Vision → Language).
        Returns: Combined results with errors if any.
        """
        try:
            # Step 1: Get Vision Agent's scene analysis
            vision_msg = {
                "type": "analyze_image",
                "image_data": image,
                "sender": "coordinator"
            }
            vision_result = self.vision_agent.process_message(vision_msg)

            if vision_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": "Vision analysis failed",
                    "details": vision_result.get("error", "Unknown")
                }

            # ✅ Extract scene description safely
            scene_analysis = vision_result.get("analysis", {}).get("scene_analysis", {})
            scene_desc = (
                scene_analysis.get("description", {}).get("natural_language", "")
            )

            if not scene_desc:
                return {
                    "status": "error",
                    "error": "No scene description available from Vision Agent"
                }

            # Step 2: Ask Language Agent to generate a caption
            caption_result = self.language_agent.generate_caption(scene_desc)

            if caption_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": "Caption generation failed",
                    "details": caption_result.get("error", "Unknown")
                }

            # ✅ Combine all results
            final_result = {
                "status": "success",
                "vision_analysis": vision_result,
                "language_caption": caption_result,
                "final_caption": caption_result.get("caption", "No caption generated"),
                "timestamp": self._get_timestamp()
            }
            return final_result

        except Exception as e:
            self.logger.error(f"Task processing error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _get_timestamp(self):
        """Helper to get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
