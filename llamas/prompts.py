def hop_1_prompts():
    return """Below is an instruction that describes a task, paired with an input that provides further context. Write a resonse that appropriately completes the request. 
    
### Instruction:
Please generate a triplet (seperated by comma) that would contribute to answering question.
    
### Input:
{}

### Response:"""

def multi_hop_prompts():
    return """Below is an instruction that describes a task, paired with an input that provides further context. Write a resonse that appropriately completes the request. 
    
### Instruction:
Given the existing triplet(s) as reasoning paths, please generate an additional triplet (seperated by comma) to extend the current path that would further contribute to answering the question.
    
### Input:
{}

### Response:"""

def relation_prompts():
    return """Below is an instruction that describes a task, paired with an input that provides further context. Write a resonse that appropriately completes the request. 

### Instruction:
Please generate all of the reasoning paths that could potentially answer the question.    

### Input:
{}
"""

def ranker_instruction_prompt():
    # TODO: Fix
    return """Given the following reasoning path(s) {} as supplementary information, is the candidate "{}" a potential answer to the given question?"""

def alpaca_prompt():
    return """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}

### Input:
{}

### Response:"""


# Ranker prompts
def ranker_prompt_initial():
    return """Assume you are a helper designed to improve predictions from a machine learning model. You will be given a question and some candidates, and you need to rank the candidates from the most possible answer to the least possible ones. You will first be given an examples so that you can get familiar with the input and output formats. If you understand your responsibility, do not output anything."""

def demonstration_prompt():
    # Change it to training example 
    return """ Below is an example.
### Instruction
The question is "What did nick clegg study at University?", and the candidate answers are [salman rushdie, liberal democrats, social anthropology, bertrand russell, political philosophy, country, master of arts, college of europe, percy bysshe shelley, buckinghamshire, university of minnesota, capitalism: the unknown ideal, chalfont st giles, united kingdom, house of commons of the united kingdom, caldicott school, robinson college, cambridge, university of cambridge, national secular society, westminster school]. Now, based on your knowledge, sort the candidates from the most likely answer to the least likely answer using the formst "[most possible answer | second possible answer | ... | least possible answer] and please start your response with "The final order:". Do not output anything except the final order. Note that your output needs to contain all the candidates in the least but not add new answers to it.

### Response 
The final order: political philosophy | social anthropology | robinson college, cambridge | salman rushdie | college of europe | percy bysshe shelley | house of commons of the united kingdom | united kingdom | university of minnesota | westminster school | capitalism: the unknown ideal | country | university of cambridge | national secular society | bertrand russell | buckinghamshire | caldicott school | chalfont st giles | liberal democrats | master of arts]
"""
