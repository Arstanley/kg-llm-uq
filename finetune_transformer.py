"""
    Finetune a sentecne transformer from the truth path
"""
from utils.graph_utils import *
from datasets import load_from_disk, load_dataset
import argparse
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses, SentenceTransformer, evaluation
import huggingface_hub
import numpy as np
import wandb
from utils.data_utils import NoDuplicatesDataLoader
from tqdm import tqdm

wandb.login()
wandb.init(project="finetune-transformer-websq")

access_token = "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"
huggingface_hub.login(token=access_token)

parser = argparse.ArgumentParser(description="parser")
parser.add_argument('--dataset_dir', default="/home/bo/Dropbox/Projects/kg-llm-uq/data")
parser.add_argument('--dataset_name', default="webqsp")

args = parser.parse_args()

finetune_data = []
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def main():
    dataset = load_from_disk(f'{args.dataset_dir}/{args.dataset_name}')
    for data_point in dataset['train']:
        graph = build_graph(data_point['graph'])
        q_entities = data_point['q_entity']
        a_entities = data_point['a_entity']
        question = data_point['question']
        
        for q_entity in q_entities:
            question = question.lower().replace(q_entity.lower().strip(), "[MASK]")

        truth_paths = get_truth_paths(q_entities, a_entities, graph)
        for truth_path in truth_paths:
            for i, _ in enumerate(truth_path):
                if i == 0:
                    finetune_data.append(InputExample(texts=[question + "? ", truth_path[0][1]]))
                else:
                    finetune_data.append(InputExample(texts=[question + "? " + "->".join([ii[1] for ii in truth_path[:i]]), truth_path[i][1]]))
            finetune_data.append(InputExample(texts=[question, "->".join([ii[1] for ii in truth_path])])) # If the current path answers the question

    # Joint training 
    dataset = load_dataset(f'rmanluo/RoG-cwq')
    for data_point in tqdm(dataset['train'], total=len(dataset['train'])):
        graph = build_graph(data_point['graph'])
        q_entities = data_point['q_entity']
        a_entities = data_point['a_entity']
        question = data_point['question']
        
        for q_entity in q_entities:
            question = question.lower().replace(q_entity.lower().strip(), "[MASK]")

        truth_paths = get_truth_paths(q_entities, a_entities, graph)
        for truth_path in truth_paths:
            for i, _ in enumerate(truth_path):
                if i == 0:
                    finetune_data.append(InputExample(texts=[question + "? ", truth_path[0][1]]))
                else:
                    finetune_data.append(InputExample(texts=[question + "? " + "->".join([ii[1] for ii in truth_path[:i]]), truth_path[i][1]]))
            finetune_data.append(InputExample(texts=[question, "->".join([ii[1] for ii in truth_path])])) # If the current path answers the question

    list1 = []
    list2 = []
    for data_point in dataset['validation']:
        graph = build_graph(data_point['graph'])
        q_entities = data_point['q_entity']
        a_entities = data_point['a_entity']
        question = data_point['question']
        truth_paths = get_truth_paths(q_entities, a_entities, graph)
        for truth_path in truth_paths:
            for i, triplet in enumerate(truth_path):
                if i == 0:
                    list1.append(question + "? ")
                    list2.append(" ".join(truth_path[0][:-1]))
                else:
                    list1.append(question + "? " + ",".join([" ".join(j) for j in truth_path[:i]]))
                    list2.append(" ".join(truth_path[i][:-1]))

    evaluator = evaluation.BinaryClassificationEvaluator(list1, list2, [1 for _ in range(len(list1))]) 
    
    train_dataloader = NoDuplicatesDataLoader(finetune_data, batch_size=32)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    num_epochs = 40
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          # evaluator=evaluator, 
          output_path='/home/bo/Dropbox/Projects/kg-llm-uq/saved_models',
          # evaluation_steps=2,
          log_callback=log_callback,
          log_steps=100)  
    
    model.save('./transformer_models')
    
    # model.save_to_hub(
    # "all-MiniLM-L6-v2-sentence-transformer-kg-llm-uq", 
    # organization="arslxx",
    # train_datasets=["arslxx/arslxx_websq"],
    # exist_ok=True, 
    # ) 

def log_callback(train_idx, epoch, training_steps, lr, loss):
    print("loss")
    wandb.log({"train loss": loss})   
    wandb.log({"epoch": epoch})
    
if __name__ == '__main__':
    main()