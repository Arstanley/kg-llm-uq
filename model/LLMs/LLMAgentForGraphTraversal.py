from .LLMAgent import LLMAgent 
from llamas.prompts import *

class LLMAgentForGraphTraversal(LLMAgent):
    """
    A class to represent a LLM graph traversing agent. It inherits the LLM Agent class.
    
    Attributes 
    ----------
    model_path: str 
        Path to the model
    device: str
        Device to load the model

    Methods
    -------
    generate(str, kwargs**) -> str:
        return the generated string from the given prompt. It also takes parameters to customize the generation.

    request_information(question: str, path: list(str)) -> str:
        Given a question, and the current traversed path. Let LLM assert what is further needed to answer the question. 
    """
    def __init__(self, model_path, device, lora_weights=None, pre_trained=False):
        super().__init__(model_path, device, lora_weights, pre_trained) 
        
    def request_information(self, question, paths):
        """ 
        Given a question, and the current traversed path. Let LLM assert what is further needed to answer the question. 

        Args:
            question (str): The incoming question 
            paths (list(str)): [(s_0, r_0, o_0), (s_1, r_1, o_1)...]
        """
        prompt = self.create_prompt(question, paths)

        return self.generate(prompt)

    def create_prompt(self, question, paths):
        """
        Create the prompt for request_information

        Args:
            question (str): see parent function 
            paths (list(str)): see parent function 

        Example:
            Input:
                question: Where is the home stadium of the team who won the 1946 World Series championship?
                Paths: [(1946 World Series, sports.sports team.championships, St. Louis Cardinals), (St. Louis Cardinals, sports.sports team.arena stadium, Roger Dean Stadium)] 
            Output:
                '''
                Question:
                    Where is the home stadium of the team who won the 1946 World Series championship?
                Supporting evidence: 
                    1946 World Series -> sports.sports team.championships -> St. Louis Cardinals,
                    St. Louis Cardinals -> sports.sports team.arena stadium -> Roger Dean Stadium

                Instruction:
                    What evidence do we further need to answer the question given the current supporting evidence? 

                Response:
                '''
        """
        if paths == None:
            return hop_1_prompts().format(question+"?")
        else: 
            concatenated_ps = '->'.join([e for triplets in paths for e in triplets[:-1]])
            question_path = f"{question}? {concatenated_ps}"    

            return multi_hop_prompts().format(question_path) 
