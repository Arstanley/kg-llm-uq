"""
    This file creates finetune data for llama finetuning
"""

import argparse
from datasets import load_from_disk
import wandb
import pickle as pkl
from tqdm import tqdm
import sys, os
import json
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils.graph_utils import *

parser = argparse.ArgumentParser(description="parser")
parser.add_argument('--dataset_dir', default="/home/bo/Dropbox/Projects/kg-llm-uq/data")
parser.add_argument('--dataset_name', default="webqsp_path_based")
parser.add_argument('--output_dir', default='/home/bo/Dropbox/Projects/kg-llm-uq/llamas')

args = parser.parse_args()

wandb.init(project="kg-llm-uq-finetune-preparation")

def main():
    ret = []

    training_data = load_from_disk(f'{args.dataset_dir}/{args.dataset_name}')['train']
    for i, data in enumerate(tqdm(training_data, total=len(training_data))):
        graph = build_graph(data['graph'])
        q_entities = data['q_entity']
        a_entities = data['a_entity']
        question = data['question']

        truth_paths = get_truth_paths(q_entities, a_entities, graph)

        # We only train on relation
        # hop1outputs = []
        # for truth_path in truth_paths:
        #     if len(truth_path) == 0:
        #         continue
            
        #     hop1outputs.append(truth_path[0][1])

        #     if len(truth_path) > 1:
        #         instruction = """Given the existing triplet(s) as reasoning paths, please generate an additional triplet (seperated by comma) to extend the current path that would further contribute to answering the question."""
        #         for j, triplet in enumerate(truth_path):
        #             if j >= 1:
        #                 cur_path = "->".join(truth_path[0])
        #                 cur_path += "->".join(["->".join(p[1:]) for p in truth_path[1:j]])
                        
        #                 input = f"""{question}? {cur_path}"""
        #                 output = '(' + ",".join(triplet) + ')'
        #                 ret.append(create_json_item(instruction, input, output))
        
        # truth_path is [(s, r, t), (s, r, t), (s, r, t)]
        paths = "\n".join(set(["->".join([path[1] for path in truth_path]) for truth_path in truth_paths]))

        instruction = """Please generate all of the reasoning paths that could potentially answer the question."""
        
        ret.append(create_json_item(instruction, question, paths))
    
    with open(f"{args.output_dir}/data/webqsp_finetune.json", "w") as f:
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

if __name__ == '__main__':
    main()