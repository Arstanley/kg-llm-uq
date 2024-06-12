from transformers import AutoTokenizer, PreTrainedTokenizerFast, LlamaForCausalLM, LlamaTokenizer, GenerationConfig, pipeline, Conversation
from peft import PeftModel, PeftConfig, PeftModelForCausalLM
import torch
from llamas.prompts import *

class LLMAgent:
    """
        A parent class to represent a LLM agent. It will be inherited by LLMs with different functionalities.
    """
    def __init__(self, model_path, device, lora_weights=None, pre_trained=False):
        self.device = device
        if pre_trained == False:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = LlamaForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=False,
                torch_dtype=torch.float16
            ).bfloat16().cuda()
        if pre_trained == True:
            print("loading pretrained model")
            assert(lora_weights!=None)
            print(lora_weights)
            config = PeftConfig.from_pretrained(lora_weights)
            self.model = LlamaForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=False,
                torch_dtype=torch.float16
            ).half().cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = PeftModelForCausalLM.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16
            ).half().cuda()
            
        if self.device == 'cuda':
            self.model = self.model.cuda()

    def generate(self, prompt, temperature=0.75, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=5, question=None, concatenated_path=None, return_output=False):
        if question != None:
            prompt = self.create_prompt(question, concatenated_path)
        inputs = self.tokenizer(prompt, return_tensors="pt") 
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature, 
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                max_new_tokens=max_new_tokens 
            ) 

        # Return logits only for the generated tokens
        s = generation_output.sequences[0]
        logits = generation_output.logits[0][0]
        scores = generation_output.scores[0][0]

        output = self.tokenizer.decode(s)
        # Whether to return full output

        if not return_output:
            return output.split('Response:')[1].strip().strip('###') # Assume the structure of the prompt have '### Response:' in the end 
        else:
            return logits, scores, generation_output 