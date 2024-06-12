from llamas.prompts import *
from .LLMAgent import LLMAgent

class LLMAgentForRelationPruning(LLMAgent):
    def __init__(self, model_path, device, lora_weights=None, pre_trained=False):
        super().__init__(model_path, device, lora_weights, pre_trained) 

    def generate_relations(self, question):
        res = self.generate(relation_prompts().format(question+"?"))
        res = '\n'.join(set(res.split('\n')))
        
        return res 
        