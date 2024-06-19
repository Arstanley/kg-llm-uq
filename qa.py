import argparse
from itertools import product
from operator import truth
from datasets import load_dataset, load_from_disk
from llamas.prompts import ranker_prompt_initial
from model.LLMs.LLMAgentForGraphTraversal import *
from model.LLMs.LLMAgentForIncontextRanking import *
from model.LLMs.LLMAgentForPostProcessRanking import *
from model.LLMs.LLMAgentForRelationPruning import *
import scipy.stats as stats

from utils.graph_utils import *
from model.cp import *
from datasets import *
from tqdm import tqdm
import pickle as pkl
import wandb
import re
import json


parser = argparse.ArgumentParser(description="parser")
parser.add_argument('--dataset_dir', default="/home/bo/Dropbox/Projects/kg-llm-uq/data")
parser.add_argument('--dataset_name', default="webqsp")
parser.add_argument('--output_path', default="/home/bo/Dropbox/Projects/kg-llm-uq/results")
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--similarity_method', default="transformer")
parser.add_argument('--generation_method', default="concatenation") # other method could be concatenation

parser.add_argument('--partition', default='test')

# LLM Args
parser.add_argument('--pre_trained', type=bool, default=False)
parser.add_argument('--lora_weights', default="/home/bo/Dropbox/Projects/kg-llm-uq/llamas/models/checkpoint-200")

# LLM Pruner
parser.add_argument('--prune', type=bool, default=False)
parser.add_argument('--lora_weights_pruner', default="/home/bo/Dropbox/Projects/kg-llm-uq/llamas/models/checkpoint-200")

# LLM Ranker
parser.add_argument('--llm_ranker', type=bool, default=False)
parser.add_argument('--lora_weights_ranker', default='/home/bo/Dropbox/Projects/kg-llm-uq/llamas/models/checkpoint-2400')
parser.add_argument('--llm_ranker_pretrained', type=bool, default=False)
parser.add_argument('--calibrate_rog', type=bool, default=False)

args = parser.parse_args()

wandb.init(project="kg-llm-uq", name=str(args.epsilon) + "_" + args.dataset_name, config=args)

epsilons = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

def split_trainset(train_set, split_rate=0.5): 
    """
        Splitting training set into training and caliberation. Default to 30% caliberation set.
        return: training set (30% of original data), and caliberation set(70% of original data)
    """
    split_datasets = train_set.train_test_split(test_size=split_rate)

    return split_datasets['train'], split_datasets['test']

def main():
    """
    Main function to run the QA model.
    """
    # Load dataset
    if os.path.isdir(f'{args.dataset_dir}/{args.dataset_name}_path_based'):
        dataset = load_from_disk(f'{args.dataset_dir}/{args.dataset_name}_path_based')
    else:
        dataset = load_dataset(f'rmanluo/RoG-{args.dataset_name}')
        # Split train set into train and caliberation
        train_dataset, caliberation_dataset = split_trainset(dataset['train'])
        # Save to disk
        dataset['train'] = train_dataset
        dataset['caliberation'] = caliberation_dataset
        dataset.save_to_disk(f'{args.dataset_dir}/{args.dataset_name}')

    # Load caliberation dataset
    # _, caliberation_dataset = split_trainset(dataset['caliberation'], split_rate=0.005)
    caliberation_dataset = dataset['caliberation']

    # Load train, test, and validation datasets
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    val_dataset = dataset['validation']

    # Initialize LLM prunem
    llm_pruner = None
    llm_ranker = None
    if args.prune == True:
        llm_pruner = LLMAgentForRelationPruning("meta-llama/Llama-2-7b-chat-hf", device="cuda", lora_weights=args.lora_weights_pruner, pre_trained=True)  
    # if args.llm_ranker == True:
        # llm_ranker = LLMAgentForPostProcessRanking("meta-llama/Meta-Llama-3-8B-Instruct", device="cuda", lora_weights=args.lora_weights_ranker, pre_trained=args.llm_ranker_pretrained)
    # Initialize LLM model
    if args.generation_method == "llm":
        llm_agent = LLMAgentForGraphTraversal("meta-llama/Llama-2-7b-chat-hf", device='cuda', lora_weights=args.lora_weights, pre_trained=args.pre_trained)
        # Create conformal predictor model using LLM
        cp_model = ConformalPredictor(caliberation_dataset, dataset=args.dataset_name, llm_agent=llm_agent, alpha=args.alpha, llm_pruner=llm_pruner)  # This creates all of the q_hats needed for conformal prediction
    # Create conformal predictor model using concatenation
    if args.generation_method == "concatenation":
        cp_model = ConformalPredictor(caliberation_dataset, dataset=args.dataset_name, alphas=[0.5, 0.5, 0.5], llm_pruner=llm_pruner, llm_ranker=llm_ranker, calibrate_with_reasoning_paths=args.calibrate_rog)
   
    # set up a separate outputfile
    output_path = f'{args.output_path}/{args.dataset_name}_{args.epsilon}_{args.partition}_predictions.jsonl'
    fout = open(output_path, "a")

    # Determine partition dataset to run o
    # if args.partition == 'validation':
    #     total_len = len(val_dataset) 
    #     run_dataset = val_dataset
    # if args.partition == 'train':
    #     total_len = len(train_dataset)
    #     run_dataset = train_dataset
    # if args.partition == 'test':
    #     total_len = len(test_dataset)
    #     run_dataset = test_dataset

    # 1 ) Get the search space for the alphas 
    alpha1 = np.arange(0.1, 1, 0.3)
    alpha2 = np.arange(0.1, 1, 0.2)
    alpha3 = np.arange(0.1, 1, 0.1)
    search_spaces = {"alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3}
    combinations = list(product(*search_spaces.values()))
    print("Alpha Space: ", len(combinations))
    
    # Load p_values from disk if it exists, otherwise calculate it and save it
    p_values_path = f"./p_values_{args.dataset_name}.json"

    p_values = {}
    if os.path.isfile(p_values_path):
        with open(p_values_path, "r") as f:
            p_values = json.load(f)
        combinations_idx = len(p_values)
    else:
        # p_values = [] # This will be a 2d array, each row will be the p_values for a combination. We include every epsilon in the p_values.
        combinations_idx = 0
        
    for i, combination in enumerate(tqdm(combinations[combinations_idx:])):
        alpha1 = combination[0]
        alpha2 = combination[1]
        alpha3 = combination[2]

        current_p_values = get_p_values(combination, cp_model, caliberation_dataset.select(range(150)), epsilons, fout) 
        p_values[i] = (current_p_values, combination)

        with open(p_values_path, "w") as f:
            json.dump(p_values, f)
        
    p_values = np.array(p_values)

    # Check if the object exist, if not, then run the code and save it.
    rejected_combinations_for_epsilon_path = f"./rejected_combinations_for_epsilon_{args.dataset_name}.pkl"
    if not os.path.isfile(rejected_combinations_for_epsilon_path):
        # Get the rejected combinations for each epsilon
        rejected_combinations_for_epsilon = []
        for i, epsilon in enumerate(epsilons):
            current_p_values = p_values[:, i]  # get the p_values for the current epsilon
            rejected_combinations_for_epsilon.append([combination for i, combination in enumerate(combinations) if current_p_values[i] < (0.05 / len(combinations))])
        
        # Save these 
        pkl.dump(rejected_combinations_for_epsilon, open(rejected_combinations_for_epsilon_path, "wb"))
    else:
        # Load the object from disk
        rejected_combinations_for_epsilon = pkl.load(open(rejected_combinations_for_epsilon_path, "rb"))
    # now in the validation set, we find the configuration that has the smallest average set 
    user_epsilon = args.epsilon 
    rejected_combinations = rejected_combinations_for_epsilon[epsilons.index(user_epsilon)]
    
    for combination in rejected_combinations:
        avg_set_size = get_avg_set_size(combination, cp_model, val_dataset.select(range(50)), fout)

        if avg_set_size < min_avg_set_size:
            min_avg_set_size = avg_set_size
            best_combination = combination
    
    # 2 ) Run on test dataset  
    cp_model.set_alpha(best_combination)
    _, avg_set_size, hits_at_1 = run_cp_model_on_dataset(test_dataset, cp_model, fout)

    print(f"Best alpha: {best_combination} with avg set size: {avg_set_size} and hits@1: {hits_at_1}, epsilon: {args.epsilon}")
     
def run_cp_model_on_dataset(dataset, cp_model, fout):
    total_length = 0
    correct = 0
    hits_at_1 = 0
    correct_all = 0 # Track all() metrics

    for i, data in enumerate(tqdm(dataset, total=len(dataset))): 
        wandb.log({"progress": (i+1)/len(dataset)})
         
        # Build graph
        graph = build_graph(data['graph'])
        # Get question entities
        q_entity = data['q_entity']
        # Get answer entities
        a_entity = data['a_entity']
        # Get question text
        question = data['question']
        # Get question id 
        question_id = data['id']
        
        try:
            final_answers = cp_model.predict(q_entity, graph, question, question_id)
            print(final_answers)

            # Create dict to store prediction
            res = {"id": data['id'], "question": question, "final_answers": final_answers, "a_entity": a_entity}
            fout.write(json.dumps(res) + "\n")
            fout.flush()

            total_length += (len(final_answers) / len(a_entity))

            answer_set = final_answers 
            correct += len([0 for a in a_entity if a.lower().strip() in answer_set]) / len(a_entity)
            hits_at_1 += any(a.lower().strip() in answer_set for a in a_entity)
             
            correct_all += all(a.lower().strip() in answer_set for a in a_entity)

        except Exception:
            raise Exception
            pass  
        
        wandb.log({"Acc": correct/(i+1), "Avg set length": total_length / (i+1), "acc_all": correct_all/(i+1), "hits@1": hits_at_1/(i+1)})
    
    return correct, total_length / len(dataset), hits_at_1 
    # train_dataset.save_to_disk("train_webqsp_0.1.hf")

def get_p_values(configuration, cp_model, calibration_set, epsilon, fout):
    # Get p values for all epsilons in question
    cp_model.set_alpha(configuration)
    _, _, hits_at_1 = run_cp_model_on_dataset(calibration_set, cp_model, fout)
    R = (len(calibration_set) - hits_at_1) # Risk level
    print(R, len(calibration_set), epsilon)

    p_values = [] 
    for epsilon in epsilons:
        p_value = stats.binom.cdf(R, len(calibration_set), epsilon)
        p_values.append(p_value)

    return p_values

def get_avg_set_size(configuration, cp_model, validation_set, fout):
    cp_model.set_alpha(configuration) 
    _, avg_set_size, _ = run_cp_model_on_dataset(validation_set, cp_model, fout)

    return avg_set_size

if __name__ == '__main__':
    main()
