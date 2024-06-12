from transformers import Conversation, pipeline
import torch
from .LLMAgent import LLMAgent

class LLMAgentForIncontextRanking(LLMAgent):
    def __init__(self, model_path, device):     
        """
        A class to represent a llm ranking agent using in-context inference. 
        """
        self.generator = pipeline(task="conversational", model=model_path, device=device, torch_dtype=torch.float16)
        self.conversation = self.generator(self.conversation)
    
    def add_initial_message(self, initial_message):
        self.conversation = Conversation(initial_message)
        self.conversation = self.generator(self.conversation)
    
    def add_user_message(self, message):
        """
        Add a message to the conversation.

        Parameter
        ---------
        message: str
            Message to add
        """
        self.conversation.add_message({"role": "user", "content": message}) 
        self.conversation = self.generator(self.conversation)
    
    def get_conversation(self):
        return self.conversation
    