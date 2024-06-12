from .LLMAgent import LLMAgent
from llamas.prompts import *
from transformers import pipeline
import torch

class LLMAgentForPostProcessRanking(LLMAgent):
    """
        This class is used for post processing ranking. Specifically, besides generation, it will get the logits for specific token. In our case, it would be the logits for "yes" token
    """
    def __init__(self, model_path, device, lora_weights=None, pre_trained=True):
        """
        Initialize the LLMAgentForPostProcessRankng class.

        Args:
            model_path (str): The path to the model.
            device (str): The device to use for computation.
            lora_weights (Optional[str]): The path to the LORA weights. Defaults to None.
            pre_trained (bool): Whether to use pre-trained weights. Defaults to True.

        Returns:
            None
        """
        self.pre_trained = pre_trained
        super().__init__(model_path, device, lora_weights, pre_trained) 
    
    def isPretrained(self):
        return self.pre_trained
        
    def get_softmax_score(self, question, candidate, paths):
        """
        Generate the logits for a given question, candidate, and paths.

        Args:
            question (str): The question to generate logits for.
            candidate (str): The candidate to generate logits for.
            paths (str): The paths to generate logits for.

        Returns:
            float: The first token logit.
        """
        if self.pre_trained == True:
            prompt = self.create_prompt(question, candidate, paths, pretrained=True)
            print(prompt)
        else:
            prompt = self.create_prompt(question, candidate, paths, pretrained=False)
        
        logits, scores, _ = self.generate(prompt, return_output=True) 
        
        # Get token id from tokenizer and find corresponding logits 
        # yes_token_id = self.tokenizer('Yes', return_tensors="pt")[1]

        yes_token = self.tokenizer("Yes")['input_ids'][1]
        no_token = self.tokenizer("No")['input_ids'][1]
        yes_logit = logits[yes_token] # hardcoding for now 9642 = 'Yes'
        no_logit = logits[no_token]
        yes_score = scores[yes_token]
        no_score = scores[no_token]

        return torch.tensor([yes_score, no_score])
        # return torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)

    def create_prompt(self, question, candidate, paths, pretrained):
        '''
            Return prompt in alpaca format
        '''
        if pretrained:
            paths_str = "[" + ",".join(paths) + "]"
            instruction = ranker_instruction_prompt().format(paths_str, candidate)
            i = 0
            while len(instruction) > 10000 and i < len(paths):
                path_str = "[" + ','.join(paths[i:]) + "]"
                instruction = ranker_instruction_prompt().format(path_str, candidate)
                i += 1
            return alpaca_prompt().format(instruction, question+"?")
        else:
        #     return f"""###Instruction: I will provide some examples and explainations. Try to understand these examples and generalize the result. For example, "80910" is not a potential answer to the question "in what state was the air force academy established?" because it is not in the list of states, but "kentucy" is a potential answer because it is a state. Another example is that the candidate "snowkit" is not a potential answer to the question "what disease did helen keller have?" because "snowkit" is not a disease. Whereas "cancer" potentially answers the question because it is a type of disease. Now, a question is given: Does the candidate "{candidate}" potentially answer the question "{question}"? Answer with 'Yes' or 'No' only. 
        # ###Response:""" 
            system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            user_message = f"""Is "{candidate}" an answer to "{question}" Answer with 'Yes' or 'No' only. DO NOT output anything else."""

            messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},]

            prompt = self.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)
            return prompt 

