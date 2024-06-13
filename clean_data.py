from collections import defaultdict
from datasets import load_from_disk, load_dataset
from utils.graph_utils import *
from transformers import pipeline
import torch
from qa import split_trainset

dataset_name = 'webqsp'

LLAMA_MODEL = 'meta-llama/Llama-2-7b-chat-hf' # Can use pretrain in the future
# generator = pipeline('text-generation', model=LLAMA_MODEL, torch_dtype=torch.float16, device=0)

def main():
    # Load data
    # data = load_from_disk(f'./data/{dataset_name}')
    try:
        data = load_from_disk(f'./data/{dataset_name}')
    except:
        dataset = load_dataset(f'rmanluo/RoG-{dataset_name}')
        # Split train set into train and caliberation
        train_dataset, caliberation_dataset = split_trainset(dataset['train'])
        # Save to disk
        dataset['train'] = train_dataset
        dataset['caliberation'] = caliberation_dataset
        dataset.save_to_disk(f'./data/{dataset_name}')

    data['train'] = data['train'].filter(clean_data).map(select_entities) 
    data['test'] = data['test'].filter(clean_data).map(select_entities)
    data['validation'] = data['validation'].filter(clean_data).map(select_entities)
    data['caliberation'] = data['caliberation'].filter(clean_data).map(select_entities)
    # data['train'] = data['train'].map(get_truth_paths)

    data.save_to_disk(f'./data/{dataset_name}_path_based')

def clean_data(data_point):
    # Remove all of the qas that does not have paths
    q_entity = data_point['q_entity']
    a_entity = data_point['a_entity']
    g = build_graph(data_point['graph'])

    truth_paths = get_truth_paths(q_entity, a_entity, g)

    if len(truth_paths) == 0:
        return False 

    return True

def select_entities(data_point):
    # Select subset of a_entity that has paths
    q_entity = data_point['q_entity']
    a_entity = data_point['a_entity']
    g = build_graph(data_point['graph'])

    truth_paths = get_truth_paths(q_entity, a_entity, g)

    a_entity_tmp = [] 
    q_entity_tmp = []
    for truth_path in truth_paths:
        if len(truth_path) == 0:
            continue
        cur_a = truth_path[-1][-1]
        cur_q = truth_path[0][0]
        a_entity_tmp.append(cur_a)
        q_entity_tmp.append(cur_q)
    
    a_entity_tmp = list(set(a_entity_tmp))
    q_entity_tmp = list(set(q_entity_tmp))

    data_point['q_entity'] = q_entity_tmp if len(q_entity_tmp) != 0 else q_entity 
    data_point['a_entity'] = a_entity_tmp if len(a_entity_tmp) != 0 else a_entity

    return data_point

def get_truth_paths_(data_point):
    # Let LLM be the strong teacher to choose the turth path 
    q_entity = data_point['q_entity']
    a_entity = data_point['a_entity']
    question = data_point['question']
    g = build_graph(data_point['graph'])

    truth_paths = get_k_simple_paths(q_entity, a_entity, g)
    
    # For every QA pair, we want to select one from the top 10 paths    
    final_paths = []
    entpair2paths = defaultdict(list)
    for path in truth_paths:
        q_ent = path[0][0]
        a_ent = path[-1][-1]
        
        if q_ent not in q_entity or a_ent not in a_entity:
            # sanity check
            raise Exception
    
        entpair2paths[q_ent + "\n" + a_ent].append(path)
        
    for key, paths in entpair2paths.items():
        q_ent, a_ent = key.split("\n")
        reasoning_path_str = ""       
        for i, path in enumerate(paths):
            reasoning_path_str += f"{(str(i+1))} "
            reasoning_path_str += "->".join("->".join([p[0], p[1]]) for p in path)
            reasoning_path_str += "->" + path[-1][-1]
            reasoning_path_str += "\n"
        prompt = path_selection_prompt().format(reasoning_path_str, question)    
        print(generator(prompt, return_full_text=False)[0]['generated_text'])
         
def path_selection_prompt():
    return """Given the following potential reasoning paths, select all of the options that provide an explaination to the question. Only output the number of your selection and DO NOT output anything else.

Reasoning paths:
{}
Question:
{}?
"""

if __name__ == '__main__':
    main()