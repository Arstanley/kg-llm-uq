import argparse
from datasets import load_from_disk, concatenate_datasets
import torch
from transformers import pipeline
from tqdm import tqdm
import wandb
from collections import defaultdict, Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity 
from itertools import product
import pickle as pkl
import os

DATASET_DIR = "/home/bo/Dropbox/Projects/kg-llm-uq/data"
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="webqsp")
parser.add_argument("--m", default=20) # Max numebr of generations

args = parser.parse_args()

# Compute the cartesian product of the search spaces
search_spaces = {"lambda1": np.arange(0, 2.5, 1), "lambda2": np.arange(0, 2.5, 1)}
combinations = list(product(*search_spaces.values()))
num_combinations = len(combinations)
print(f"Number of combinations: {num_combinations}")
print(f"lambda1: {search_spaces['lambda1']}")
print(f"lambda2: {search_spaces['lambda2']}")

# combinations = [(0.5, 0.5)]

model = SentenceTransformer("all-MiniLM-L6-v2") # Probably shouldn't be a global variable - gotta move fast
text_generator = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", batch_size=8, device=0, torch_dtype=torch.float16)
text_generator.tokenizer.pad_token_id = text_generator.model.config.eos_token_id
def main(alpha):
    # Load data 
    dataset = load_from_disk(f'{DATASET_DIR}/{args.dataset}_path_based')
    dataset['calibration'] = concatenate_datasets([dataset['train'], dataset['caliberation']])

    # Now calibrate on the data / find the optimal parameter 
    final_res = 0
    final_lambda1, final_lambda2 = None, None 

    print("Start exploring hyperparameters") 
    for lambda1, lambda2 in tqdm(combinations, total=len(combinations)):
        print(f"Current hyper parameters --- lambda1: {lambda1}, lambda2: {lambda2}")
        # Use the lambda1, lambda2 values in your computation
        q_hat = calibrate(dataset['calibration'], lambda1, lambda2, alpha)
        res, _ = evaluate(dataset['validation'], q_hat, lambda1, lambda2)
        if res > final_res:
            final_res = res 
            final_lambda1, final_lambda2 = lambda1, lambda2 
        print("Current hits@1: ", res, "Best hits@1: ", final_res)

    print(f"Best lambda1, lambda2: {final_lambda1}, {final_lambda2} with val score: {final_res}")
    test_res, avg_ss = evaluate(dataset['test'], q_hat, lambda1, lambda2)

    print(f"test score: {test_res}, avg_ss: {avg_ss}")

def calibrate(cal_dataset, lambda1, lambda2, alpha):
    # Dataset should be calibration 
    non_conformity_scores = []
    if not os.path.isfile(f'calibrated_scores/{args.dataset}_{lambda1}_{lambda2}_scores.pkl'):
        for i, data in enumerate(tqdm(cal_dataset, total=len(cal_dataset))):
            question = data['question']  
            a_entities = data['a_entity']

            freq_counter = defaultdict(int)
            for j in range(args.m):
                response = text_generator(question + "?", temperature=1, max_length=60, num_return_sequences=1)[0]['generated_text'] # TO BE IMPLEMENTED
                freq_counter[response] += 1

            for a_entity in a_entities:
                non_conformity_score = freq_counter[a_entity] / args.m + lambda1 * compute_normalized_entropy(freq_counter.keys()) - lambda2 * oned_cosine_similarity(model.encode(a_entity), model.encode(max(freq_counter, key=freq_counter.get)))
                non_conformity_scores.append(non_conformity_score) 

            with open(f'calibrated_scores/{args.dataset}_{lambda1}_{lambda2}_scores.pkl', 'wb') as f:
                pkl.dump(non_conformity_scores, f)
    else:
        with open(f'calibrated_scores/{args.dataset}_{lambda1}_{lambda2}_scores.pkl', 'rb') as f:
            non_conformity_scores = pkl.load(f)
            if len(non_conformity_scores) < len(cal_dataset):
                cal_dataset = cal_dataset.select(range(len(non_conformity_scores), len(cal_dataset)))
                def cal_data():
                    for data in cal_dataset.select(range(1)):
                        for _ in range(args.m):
                            yield data['question'] + '?' 

                freq_counter = defaultdict(int)
                for i, result in enumerate(tqdm(text_generator(cal_data(), temperature=1, max_length=60, num_return_sequences=1), total=len(cal_dataset) * args.m)):
                    if i % args.m == 0:
                        freq_counter = defaultdict(int)
                    data = cal_dataset[i // args.m - 1]
                    response = result[0]['generated_text']
                    question = data['question']  
                    a_entities = data['a_entity']
                    freq_counter[response] += 1
                    if i % args.m == args.m - 1:
                        for a_entity in a_entities:
                            non_conformity_score = freq_counter[a_entity] / args.m + lambda1 * compute_normalized_entropy(freq_counter.keys()) - lambda2 * oned_cosine_similarity(model.encode(a_entity), model.encode(max(freq_counter, key=freq_counter.get)))
                            non_conformity_scores.append(non_conformity_score) 

                    # freq_counter = defaultdict(int)
                    # for j in range(args.m):
                    #     response = text_generator(question + "?", temperature=1, max_length=60, num_return_sequences=1)[0]['generated_text'] # TO BE IMPLEMENTED
                    #     freq_counter[response] += 1

                    # for a_entity in a_entities:
                    #     non_conformity_score = freq_counter[a_entity] / args.m + lambda1 * compute_normalized_entropy(freq_counter.keys()) - lambda2 * oned_cosine_similarity(model.encode(a_entity), model.encode(max(freq_counter, key=freq_counter.get)))
                    #     non_conformity_scores.append(non_conformity_score) 
                with open(f'calibrated_scores/{args.dataset}_{lambda1}_{lambda2}_scores.pkl', 'wb') as f:
                    pkl.dump(non_conformity_scores, f)
            else:
                print(f'Skipping calibration for {args.dataset} with lambda1={lambda1}, lambda2={lambda2} because it is already saved.')

    n = len(non_conformity_scores)
    q_hat = np.quantile(non_conformity_scores, ((n+1) * (1-float(alpha)) )/n)
    return q_hat 

def word_in_list_of_sentences(word, list_of_sentences):
    word_lower = word.lower().strip()
    return any(word_lower in sentence.lower() for sentence in list_of_sentences)
    
def evaluate(dataset, q_hat, lambda1, lambda2):
    """
    Evaluates the performance of a model on a given dataset.

    Args:
        dataset (list): The dataset to evaluate the model on.
        q_hat (float): The threshold value for non-conformity score.
        lambda1 (float): The weight for normalized entropy.
        lambda2 (float): The weight for cosine similarity.

    Returns:
        float: The accuracy of the model on the dataset.
    """
    # Dataset should be val or test
    hits_at_1 = 0
    total_ss = 0
    def test_data():
        print(len(dataset))
        for data in dataset:
            for _ in range(args.m):
                yield data['question'] + '?' 

    freq_counter = defaultdict(int)
    for i, result in enumerate(tqdm(text_generator(test_data(), temperature=1, max_length=60, num_return_sequences=1), total=len(dataset) * args.m)):
        if i % args.m == 0:
            freq_counter = defaultdict(int)
        data = dataset[i // args.m - 1]
        response = result[0]['generated_text']
        a_entities = data['a_entity']

        freq_counter[response] += 1

        if i % args.m == args.m - 1:
            max_key = max(freq_counter, key=lambda k: freq_counter[k]) 
            ret = [] 
            for response in freq_counter.keys(): 
                non_conformity_score = freq_counter[response] / args.m + lambda1 * compute_normalized_entropy(freq_counter.keys()) - lambda2 * oned_cosine_similarity(model.encode(response), model.encode(max_key))
                if non_conformity_score < q_hat:
                    ret.append(response)     
            
            total_ss += len(ret)/len(a_entities) 
            for a_entity in a_entities:
                if word_in_list_of_sentences(a_entity, ret):
                    hits_at_1 += 1
        
    return hits_at_1 / len(dataset), total_ss / len(dataset)

def oned_cosine_similarity(a, b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]

def compute_normalized_entropy(responses):
    # Step 1: Count the frequency of each unique response
    response_counts = Counter(responses)
    
    # Step 2: Calculate the probabilities of each unique response
    total_responses = sum(response_counts.values())
    probabilities = np.array([count / total_responses for count in response_counts.values()])
    
    # Step 3: Compute the entropy
    entropy = -np.sum(probabilities * np.log(probabilities))
    
    # Step 4: Normalize the entropy
    num_unique_responses = len(response_counts)
    if num_unique_responses > 1:
        normalized_entropy = entropy / np.log(num_unique_responses)
    else:
        normalized_entropy = 0  # Entropy is zero if there's only one unique response
    
    return normalized_entropy

if __name__ == "__main__":
    main(0.5)
    main(0.4)
    main(0.3)
    main(0.2)
    main(0.15)
    main(0.1)