import json 
import argparse
import pandas as pd
import sys, os
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils.graph_utils import *


# def load_data(file_path):
#     return json.loads(file_path)
def split_trainset(train_set, split_rate=0.5): 
    """
        Splitting training set into training and caliberation. Default to 30% caliberation set.
        return: training set (30% of original data), and caliberation set(70% of original data)
    """
    split_datasets = train_set.train_test_split(test_size=split_rate)

    return split_datasets['train'], split_datasets['test']


def main(args):
    # parser   
    if os.path.isdir(f'{args.dataset_dir}/{args.dataset_name}'):
        dataset = load_from_disk(f'{args.dataset_dir}/{args.dataset_name}')
    else:
        dataset = load_dataset(f'rmanluo/RoG-{args.dataset_name}')
        # Split train set into train and caliberation
        train_dataset, caliberation_dataset = split_trainset(dataset['train'])
        # Save to disk
        dataset['train'] = train_dataset
        dataset['caliberation'] = caliberation_dataset
        dataset.save_to_disk(f'{args.dataset_dir}/{args.dataset_name}')
    
    trainset = dataset['train']

    result_data = []
    splits = args.split
    for split in splits:
        input_path = f'{args.file_path}/webqsp_path_based_0.1_train_predictions.jsonl'
        with open(input_path, "r") as f:
            for line in f:
                result_data.append(json.loads(line))

    ret = []

    id_to_entry = {entry['id']: entry for entry in trainset}

    for data_point in tqdm(result_data):
        id = data_point['id']
        data_entry = id_to_entry[id] 
        
        question = data_entry['question']
        answer_set = data_point['answer_set']
        reasoning_paths = data_point['reasoning_paths']

        graph = build_graph(data_entry["graph"])
        paths = get_truth_paths(data_entry["q_entity"], data_entry["a_entity"], graph)
        # Let's also get the ground truth relation
        ground_relations = set()
        for path in paths:
            ground_relations.add(tuple([p[1] for p in path])) 
        
        ground_relations = list(ground_relations) 
        if len(ground_relations) > 0:
            ground_truth_paths = apply_rules(graph, ground_relations, data_entry['q_entity'])
            list_of_gt_paths = [path_to_string(p) for p in ground_truth_paths]
        
        a_entity_lower_stripped = [a.lower().strip() for a in data_point['a_entity']]

        ground_truth_paths_str = "[" + ",".join(list_of_gt_paths) + "]"
        
        for a_entity in a_entity_lower_stripped:
            instruction = f"""Given the following reasoning path(s) {ground_truth_paths_str} as supplementary information, is the candidate [{a_entity}] a potential answer to the given question?"""
            input = question + "?"
            output = "Yes\n" 
            ret.append(create_json_item(instruction, input, output))         

            if len(answer_set) == 0:
                continue
            answer = random.choice(answer_set).lower().strip()
            while answer in a_entity_lower_stripped: 
                answer = random.choice(answer_set).lower().strip()
            instruction = f"""Given the following reasoning path(s) {ground_truth_paths_str} as supplementary information, is the candidate [{answer}] a potential answer to the given question?"""
            input = question + "?"
            output = "Yes\n" if answer in a_entity_lower_stripped else "No\n"
            ret.append(create_json_item(instruction, input, output))         
            
    with open(f"/home/bo/Dropbox/Projects/kg-llm-uq/llamas/data/{args.output_filename}", "w") as f:
        json.dump(ret, f, indent=4) 
    
    return ret
    
def create_json_item(instruction, input, output):
    """Create an item for json

    Args:
        instruction (str)
        input (str)
        output (str)
    
    Returns:
        dict
    """
    ret = {}
    ret['Instruction'] = instruction
    ret['Input'] = input
    ret['Output'] = output
    
    return ret        

def apply_rules(graph, rules, srouce_entities):
    results = []
    for entity in srouce_entities:
        for rule in rules:
            res = bfs_with_rule(graph, entity, rule)
            results.extend(res)
    return results

def path_to_string(path: list) -> str:
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"
            
    return result.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser') 
    parser.add_argument('--file_path', default="/home/bo/Dropbox/Projects/kg-llm-uq/results") # File that we want to convert to pretraining data
    parser.add_argument('--split', nargs="+", default='train')
    parser.add_argument('--output_filename', default="webqsp_finetune_ranking.json")
    parser.add_argument('--dataset_dir', default="/home/bo/Dropbox/Projects/kg-llm-uq/data")
    parser.add_argument('--dataset_name', default="webqsp_path_based")

    args = parser.parse_args()
    main(args)
