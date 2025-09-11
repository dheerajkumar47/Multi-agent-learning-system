# agents/base_agent.py
import logging
import json
from datetime import datetime
from typing import Dict, Any, List
class BaseAgent:
    """
    Base class that all agents inherit from.
    Handles common functionality like messaging, logging, and state management.
    """
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.message_history = []
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"Agent_{name}")
        self.logger.info(f"Agent {name} initialized with capabilities: {capabilities}")
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method that processes incoming messages.
        Each agent type will override this method.
        """
        start_time = datetime.now()
        
        try:
            # Log incoming message
            self.logger.info(f"Processing message: {message.get('type', 'unknown')}")
            
            # Store message in history
            self.message_history.append({
                "timestamp": start_time,
                "type": "received",
                "content": message
            })
            
            # Process the message (implemented by child classes)
            response = self._handle_message(message)
            
            # Calculate response time
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Update performance stats
            self.performance_stats["total_requests"] += 1
            self.performance_stats["successful_requests"] += 1
            self.performance_stats["average_response_time"] = (
                (self.performance_stats["average_response_time"] * (self.performance_stats["total_requests"] - 1) + response_time) 
                / self.performance_stats["total_requests"]
            )
            
            # Store response in history
            self.message_history.append({
                "timestamp": end_time,
                "type": "sent",
                "content": response,
                "response_time": response_time
            })
            
            self.logger.info(f"Message processed successfully in {response_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "agent": self.name
            }
    
    def _handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override this method in child classes to implement specific behavior.
        """
        return {
            "status": "success",
            "response": f"Echo from {self.name}: {message}",
            "agent": self.name,
            "capabilities": self.capabilities
        }
    
    def send_message_to(self, target_agent, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to another agent and return the response.
        """
        self.logger.info(f"Sending message to {target_agent.name}")
        return target_agent.process_message(message)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Return current agent status and performance metrics.
        """
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "status": "online",
            "performance": self.performance_stats,
            "message_count": len(self.message_history)
        }
    
    def clear_history(self):
        """
        Clear message history (useful for testing).
        """
        self.message_history = []
        self.logger.info("Message history cleared")