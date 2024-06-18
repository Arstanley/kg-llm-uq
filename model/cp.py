import copy
from sklearn.metrics.pairwise import cosine_similarity 
import sys
import os
from tqdm import tqdm
import heapq
from collections import defaultdict 
import random
import torch
from accelerate.utils import gather_object
from accelerate import PartialState

from torch.nn.parallel import DistributedDataParallel as DDP

from sentence_transformers import SentenceTransformer
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import graph_utils
from utils.format_utils import *
from utils.data_utils import * 
from llamas.prompts import *
import wandb
import re
from transformers import pipeline, AutoTokenizer, LlamaForCausalLM

import numpy as np

# text_generator = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", batch_size=4, device=0, torch_dtype=torch.float16)
# text_generator.tokenizer.pad_token_id = text_generator.model.config.eos_token_id

distributed_state = PartialState()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map=distributed_state.device
).bfloat16()

def batch_data(data_generator, batch_size):
    # batches = []
    # batch_answers = []
    batch = []
    answers = []
    for sentence, answer in data_generator:
        batch.append(sentence)
        answers.append(answer)
        if len(batch) == batch_size:
            yield batch, answers
            batch = []
            answers = []
    # return zip(batches, batch_answers)

    if batch:
        yield batch, answers

def generate_with_logits(model, tokenizer, batch, temperature=1, max_new_tokens=5):
    inputs = tokenizer(batch, return_tensors='pt', padding=True)
    inputs = {key: value.to(distributed_state.device) for key, value in inputs.items()}  # Ensure inputs are on GPU if available
    
    output = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        temperature=temperature, 
        max_new_tokens=max_new_tokens, 
        num_return_sequences=1, 
        output_scores=True, 
        return_dict_in_generate=True
    )

    generated_tokens = output.sequences
    logits = output['scores']

    generated_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_tokens]
    return generated_texts, logits 

class ConformalPredictor:
    """
    Top class for conformal prediction
    
    Attributes
    -----------
    llm_agent : LLMAgent class
        Used for LLM based KG traversal. 
    caliberation_set : Dataset
        Huggingface dataset object that contains the hold-out caliberation data for conformal prediction
    alpha : float
        Confidencee interval for conformal prediction
    encoding_method : str
        encoding method for text matching. Supports transformer and tfidf
    max_hop: int
        hyperparam that controls the reasoning path expansion and limits the search space
    """
    def __init__(self, caliberation_set, dataset='webqsp', llm_agent=None, llm_pruner=None, llm_ranker=None, alphas=[0.5, 0.5, 0.5], encoding_method="transformer", calibrate_with_reasoning_paths=False, max_hop=3):
        self.caliberation_set = caliberation_set
        self.dataset = dataset
        self.llm_agent = llm_agent
        self.llm_pruner = llm_pruner
        self.ranker = llm_ranker
        self.calibrate_with_reasoing_paths = calibrate_with_reasoning_paths
        if "webqsp"in self.dataset:
            self.max_hop=2
        if "cwq" in self.dataset:
            self.max_hop=4
        # We have 3 alphas to calibrate 
        self.alphas = alphas

        self.path_alpha = self.alphas[0]
        self.ans_alpha = self.alphas[1]
        self.post_alpha = self.alphas[2]

        print(self.path_alpha)

        # self.alpha, self.path_alpha = self.calculate_alpha(alpha)
        self.pattern = r'(m\.\d+|g\.\d+)'
        self.encoding_method = encoding_method
        self.rog_predictions_train, self.rog_predictions_test = self._rog_predictions(self.dataset)
        self._init_model()
        self._init_scores()
        self.q_hats = [np.quantile(self.path_scores[ii], ((len(self.path_scores[ii]) + 1) * (1 - self.path_alpha)) / len(self.path_scores[ii])) for ii in range(self.max_hop)]
        self.q_hats_a = np.quantile(self.ans_scores, ((len(self.ans_scores) + 1) * (1 - self.ans_alpha)) / len(self.ans_scores))
        self.q_hats_post_rank = np.quantile(self.post_rank_score, ((len(self.post_rank_score) + 1) * (1 - self.post_alpha)) / len(self.post_rank_score))

    def set_alpha(self, alphas):
        self.alphas = alphas
        self.path_alpha = self.alphas[0]
        self.ans_alpha = self.alpha[1]
        self.post_alpha = self.alpha[2]

        self.q_hats = [np.quantile(self.path_scores[ii], ((len(self.path_scores[ii]) + 1) * (1 - self.path_alpha)) / len(self.path_scores[ii])) for ii in range(self.max_hop)]
        self.q_hats_a = np.quantile(self.ans_scores, ((len(self.ans_scores) + 1) * (1 - self.ans_alpha)) / len(self.ans_scores))
        self.q_hats_post_rank = np.quantile(self.post_rank_score, ((len(self.post_rank_score) + 1) * (1 - self.post_alpha)) / len(self.post_rank_score))

    def calculate_alpha(self, alpha): 
        if self.dataset=='webqsp':
            alpha = self.alpha / 3 
            path_alpha = alpha / 2 
        if self.dataset=='cwq':
            alpha = self.alpah / 3 
            path_alpha = alpha / 4
        return alpha, path_alpha 

    def _rog_predictions(self, dataset):
        test = {}
        train = {} # Being lazy, they can just be one 
        with open(f"/home/bo/Dropbox/Projects/kg-llm-uq/baselines/reasoning-on-graphs/results/gen_rule_path/RoG-{dataset}/RoG/test/predictions_3_False.jsonl", 'r') as jsonl_file:
            json_list = list(jsonl_file)
            for line in json_list:
                entry = json.loads(line)
                train[entry['id']] = entry
                test[entry['id']] = entry 
        with open(f"/home/bo/Dropbox/Projects/kg-llm-uq/baselines/reasoning-on-graphs/results/gen_rule_path/RoG-{dataset}/RoG/train/predictions_3_False.jsonl", 'r') as jsonl_file:
            json_list = list(jsonl_file)
            for line in json_list:
                entry = json.loads(line)
                train[entry['id']] = entry 
                test[entry['id']] = entry
        with open(f"/home/bo/Dropbox/Projects/kg-llm-uq/baselines/reasoning-on-graphs/results/gen_rule_path/RoG-{dataset}/RoG/validation/predictions_3_False.jsonl", 'r') as jsonl_file:
            json_list = list(jsonl_file)
            for line in json_list:
                entry = json.loads(line)
                train[entry['id']] = entry 
                test[entry['id']] = entry
    
        return train, test

    def _init_model(self):
        if self.encoding_method == 'transformer':
            self.model = SentenceTransformer('/home/bo/Dropbox/Projects/kg-llm-uq/transformer_models')
            # self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif self.encoding_method == "tfidf":
            self.model = pretrain_tfidf()

    def _init_scores(self):
        if self.calibrate_with_reasoing_paths == False:
            scores_path = f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_scores_path.pkl'
            scores_a = f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_scores_a.pkl'
            scores_candidate = f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_scores_candidate.pkl'
            if (os.path.isfile(scores_path) and os.path.isfile(scores_a) and os.path.isfile(scores_candidate)):
                self.path_scores = pkl.load(open(scores_path, 'rb'))
                self.ans_scores = pkl.load(open(scores_a, 'rb'))
                self.post_rank_score = pkl.load(open(scores_candidate, 'rb'))
            else:
                self.path_scores, self.ans_scores, self.post_rank_score = self.calculate_scores()
        else:
            scores_path = f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_False_scores_path_rog.pkl'
            scores_a = f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_False_scores_a_rog.pkl'
            scores_candidate = f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_False_scores_candidate_rog.pkl'
            if (os.path.isfile(scores_path) and os.path.isfile(scores_a) and os.path.isfile(scores_candidate)):
                self.path_scores = pkl.load(open(scores_path, 'rb'))
                self.ans_scores = pkl.load(open(scores_a, 'rb'))
                self.post_rank_score = pkl.load(open(scores_candidate, 'rb'))
            else:
                self.path_scores, self.ans_scores, self.post_rank_score = self.calculate_scores()


    def set_alpha(self, alpha):
        """
            Set alpha. The reason that we have this is because sometimes we load the model 
            directly from pickled object and would like to change the confidence level 
        """
        self.alpha = alpha
    
    def create_tgt_sentences(self, neighbors, g, node, parent=None):
        """Helper function - create neighbors for string matching based on different mode
        
        Parameters
        ----------
        neighbors: list(str) 
            list of neighbors of the current node 
        g: NetworkX graph
            knowledge graph from the dataset
        node: str
            node_entity
        """
        if self.llm_agent == None:
            return [g[node][neig]['relation'] for neig in neighbors if neig != parent] 
        else:
            ret = []
            for neig in neighbors:
                r = g[node][neig]['relation']
                if parent != neig:
                    ret.append(f'({node}, {r}, {neig})') 
            return ret 
   
    def predict(self, q_entities, g, question, question_id):
        """Given the question, return a conformal set corresponding to the given alpha
           
        Parameters
        ----------
        q_entities: list(str)
            entities in question
        g: NetworkX Graph
            knowledge graph from the dataset
        question: str
            question in natural language
        """
        # For the initial prediction, we will directly use text similarity without llm agent 
        # The reasoning path will be a collection of paths ((s -> r -> o)...)
        # Post processing
        candidates, reasoning_paths, rog_paths = self.retrieve_candidates(q_entities, g, question, question_id)
        final_answer = self.post_process_answers(candidates, reasoning_paths, question, rog_paths)

        return final_answer 
    
    def retrieve_candidates(self, q_entities, g, question, question_id):
        reasoning_paths = []
        answers = set()

        masked_question = question
        for q_e in q_entities:
            masked_question = masked_question.lower().replace(q_e.lower().strip(), "[MASK]")

        if self.llm_pruner != None:
            expected_reasoning_paths = self.llm_pruner.generate_relations(question)
            expected_paths = expected_reasoning_paths.split("\n") 
            
        for q_entity in q_entities:
            neighbors = [n for n in g.neighbors(q_entity)]
            tgt_sentences = self.create_tgt_sentences(neighbors, g, q_entity) # Sentences to match 
            a_scores = []
            if self.llm_agent != None:
                response = self.llm_agent.request_information(question, None)
                scores = self.calculate_score(tgt_sentences, response).squeeze() # TODO: Refactor these LLM if logics to another function
            else:
                scores = self.calculate_score(tgt_sentences, masked_question + "? ").squeeze() # Text similariy score/ Has size of (1, #neighbors)
                scores = scores.reshape((1,)) if len(scores.shape) == 0 else scores

            # Next, we get the conformal prediction set 
            # First, get the caliberation threshold q_hat
            q_hat = self.q_hats[0] 
            
            cur_cp_set = []
            for i, neighbor in enumerate(neighbors):
                r = g[q_entity][neighbor]['relation']
                if scores[i] <= self.q_hats_a:
                    answers.add(neighbor.lower().strip())
                    reasoning_paths.append(f"{q_entity} -> {r} -> {neighbor}")
                    if random.uniform(0, 1) < self.path_alpha * (1-self.path_alpha): # Cool trick to limit search space. See proof in paper.
                        continue 
                if scores[i] <= q_hat:
                    heapq.heappush(cur_cp_set, (1, [neighbor, [(q_entity, r, neighbor)]])) # Cur depth is 1
                    # reasoning_paths.append(f"{q_entity} -> {r} -> {neighbor}")
            
            while len(cur_cp_set) > 0:
                (cur_depth, (next_visit, cur_path)) = heapq.heappop(cur_cp_set)
                if cur_depth >= self.max_hop:
                    break
                
                if self.llm_agent != None:
                    answer = self.llm_agent.request_information(question, cur_path) # Let the language model generate what information is further needed
                else:
                    answer = masked_question + "? " + "->".join([ii[1] for ii in cur_path]) # Concatenation - Naive method for POC 

                parent = cur_path[-1][0] # Immediate parent will be first element of the last triplet

                neighbors = self.create_tgt_sentences(g.neighbors(next_visit), g, next_visit, parent)
                
                if len(neighbors) == 0:
                    continue
                
                scores = self.calculate_score(neighbors, answer) # Calculate cosine similarity
                a_score = self.calculate_pairwise_score(masked_question + "? ", "->".join([ii[1] for ii in cur_path]))
                
                q_hat = self.q_hats[cur_depth] 

                visiting_neighbor = [n for n in g.neighbors(next_visit) if n.strip() != parent.strip()]

                continue_exploration = False
                for i, neighbor in enumerate(visiting_neighbor):
                    r = g[next_visit][neighbor]['relation']
                    if a_score <= self.q_hats_a:
                        answers.add(neighbor.lower().strip()) 
                        a_path = copy.deepcopy(cur_path)
                        a_path.append((next_visit, r, neighbor))
                        reasoning_paths.append(format_path(a_path))
                        if random.uniform(0, 1) < self.path_alpha * (1-self.path_alpha): # Cool trick to limit search space. See proof in paper.
                            continue 
                    if scores[0][i] <= q_hat:
                        continue_exploration = True
                        tmp = copy.deepcopy(cur_path)
                        tmp.append((next_visit, r, neighbor))
                        heapq.heappush(cur_cp_set, (cur_depth + 1, [neighbor, tmp]))
                # if not continue_exploration:
                #     # update reasoning path with the current path
                #     reasoning_paths.append(format_path(cur_path))
             
            rog_prediction = self.rog_predictions_test[question_id]
            rules = rog_prediction['prediction'] 
            if len(rules) > 0:
                rog_reasoning_paths = self.apply_rules(g, rules, q_entities)
                list_of_paths = [self.format_path(p) for p in rog_reasoning_paths]
            # reasoning_paths += list_of_paths

        return {s for s in answers if not re.search(self.pattern, s)}, reasoning_paths, list_of_paths

    
    def post_process_answers(self, answers, reasoning_paths, question, rog_paths):
        def cal_data():
            prompts = []
            prompt_answers = []
            for answer in answers:
                reasoning_path = rog_paths + [p for p in reasoning_paths if p.split("->")[-1].lower().strip() == answer.lower().strip()] 
                paths_str = "[" + ",".join(reasoning_path)  + "]"
                i = 0

                system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                user_message = f"""Is "{answer}" an answer to "{question}?" given the following reasoning paths: {paths_str} Answer with 'Yes' or 'No' only. DO NOT output anything else."""
                while len(user_message) > 10000 and i < len(reasoning_path):
                    path_str = "[" + ','.join(reasoning_path[i:]) + "]"
                    user_message = f"""Is "{answer}" an answer to "{question}?" given the following reasoning paths: {path_str} Answer with 'Yes' or 'No' only. DO NOT output anything else."""
                    i += 1

                messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}]

                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                prompts.append(prompt)
                prompt_answers.append(answer)

                yield prompts, answers
        def select_answers_with_no_logit_below_threshold(no_logit, batch_answers, q_hat):
            # Ensure no_logit and batch_answers are of the same length
            assert len(no_logit) == len(batch_answers), "no_logit and batch_answers must have the same length"

            # Select answers where no_logit is less than q_hat
            selected_answers = [answer for logit, answer in zip(no_logit, batch_answers) if logit < q_hat]
            return selected_answers

        final_answer = []
        batch_data_generator = batch_data(cal_data(), batch_size=32)
        all_final_answers = []

        batches = []
        batches_answers = []
        for batch, batch_answers in batch_data_generator:
            batches.append(batch) 
            batches_answers.append(batch_answers)

        data_inputs = list(zip(batches, batches_answers))
        with distributed_state.split_between_processes(data_inputs, apply_padding=True) as batched_inputs:
            for batch, batch_answers in batched_inputs:
                print(batch)
                print(batch_answers)
                # Here the logits will have shape (max_generation_length, batch_size, vocab_size)
                generated_text, logits = generate_with_logits(model, tokenizer, batch)
                first_token_logit = logits[0]
                yes_token = tokenizer("Yes")['input_ids'][1]
                no_token = tokenizer("No")['input_ids'][1]
                yes_logit = first_token_logit[:, yes_token]
                no_logit = first_token_logit[:, no_token]
                print(batch_answers)

                selected_answers = select_answers_with_no_logit_below_threshold(no_logit, batch_answers, self.q_hats_post_rank)

                all_final_answers.append(selected_answers)

            print(all_final_answers)
        # Gather all the results from all processes
        final_answer = gather_object(all_final_answers)
        return final_answer

    def calculate_score(self, tgt_sentences, cur_sentence):

        """
        Calculate the text matching score between the current entity and its neighbors  

        Parameters
        ---------- 
            sentences: list(str)
                target sentences that we are matching 
            cur_sentences: str
                source sentence that we are matching 

        Returns:
            np.array: of shape (1, len(tgt_sentences)) each entry is a score. Smaller score indicates better match
        """
        if self.encoding_method == 'tfidf':
            emb2 = self.model.fit_transform(tgt_sentences)
            emb1 = self.model.transform([cur_sentence])
        if self.encoding_method == 'transformer':
            sentences = [cur_sentence] + tgt_sentences 
            embs = self.model.encode(sentences) 
            emb1, emb2 = embs[0].reshape(1, -1), embs[1:]

        return -cosine_similarity(emb1, emb2)
    
    def calculate_scores(self):
        """
        Calculate q_hats 

        Returns:
            list(float)-length=maxhop
        """
        # scores_a = defaultdict(list) # Value to determine if, for each hop, the current exploration answers the question
        scores_a = []
        scores_path = defaultdict(list) # Scores for q_hat calculation. Each key represents each level of hops
        scores_candidate = []

        for i, data in enumerate(tqdm(self.caliberation_set, total=len(self.caliberation_set))): 
            wandb.log({"Caliberation progress": i / len(self.caliberation_set)})
            q_entities = data['q_entity']
            a_entities = data['a_entity']
            q_id = data['id']
            g = graph_utils.build_graph(data['graph'])
            question = data['question']
            masked_question = question

            if self.llm_pruner != None:
                expected_reasoning_paths = self.llm_pruner.generate_relations(question)
                expected_paths = expected_reasoning_paths.split("\n")

            for q_e in q_entities:
                masked_question = masked_question.lower().replace(q_e.lower().strip(), "[MASK]")
            
            truth_paths = graph_utils.get_truth_paths(q_entities, a_entities, g)
            if self.llm_agent != None:
                hop0answer = self.llm_agent.request_information(question, None) # Speed up. Don't need to prompt for every neighbor 
            for path in truth_paths:
                concatenated_path = []
                for hop_idx, (s, r, t) in enumerate(path):
                    if hop_idx == (len(path) - 1):
                        # Last index
                        scores_a.append(self.calculate_pairwise_score(masked_question+"? ", "->".join(ii[1] for ii in path)).item())
                    if s in q_entities and hop_idx == 0: # initial step; additional condition just for clairty
                        concatenated_path.append((s, r, t))
                        # Calculate score
                        if self.llm_agent == None:  
                            score = self.calculate_pairwise_score(masked_question + "? ", r)
                        else:
                            # answer = self.llm_agent.request_information(question, None)
                            print(hop0answer, f'({s}, {r}, {t})')
                            score = self.calculate_pairwise_score(hop0answer, f'({s}, {r}, {t})')
                        scores_path[hop_idx].append(score.item())
                    else:
                        # Calculate score
                        # Prompt will not actually be used
                        if self.llm_agent != None:
                            answer = self.llm_agent.request_information(question, concatenated_path)
                            score = self.calculate_pairwise_score(answer, f'({s}, {r}, {t})')
                        else: # concatenation
                            score = self.calculate_pairwise_score(masked_question + "? " + "->".join([ii[1] for ii in path[:hop_idx]]), r) # POC
                        concatenated_path.append((s, r, t))
                        scores_path[hop_idx].append(score.item())

            # Create the reasoning path based on RAG
            # TODO: Make predictions based on RoG 
            paths = graph_utils.get_truth_paths(q_entities, a_entities, g)
            # Let's also get the ground truth relation
            ground_relations = set()
            for path in paths:
                ground_relations.add(tuple([p[1] for p in path])) 
            
            ground_relations = list(ground_relations) 
            if len(ground_relations) > 0:
                ground_truth_paths = self.apply_rules(g, ground_relations, q_entities)
                list_of_gt_paths = [path_to_string(p) for p in ground_truth_paths]

            rog_prediction = self.rog_predictions_train[q_id]
            rules = rog_prediction['prediction'] 
            if len(rules) > 0:
                rog_reasoning_paths = self.apply_rules(g, rules, q_entities)
                list_of_paths = [self.format_path(p) for p in rog_reasoning_paths]
            list_of_gt_paths += list_of_paths

            # In additional to the gt paths, let's also add the predicted paths from ROG for better claibration

            # Now, we will use the pre-trained llm as the logit calculator 
            _, logits = generate_with_logits(model, tokenizer, a_entities)
            # logits is of shape (max_len, len(a_entities) aka batch size, vocab_size)
            no_token = tokenizer("No")['input_ids'][1]
            
            no_logit = logits[0, :, no_token]
            scores_candidate.extend(no_logit.tolist())

        self.max_hop = len(scores_path) if len(scores_path) < self.max_hop else self.max_hop
        if self.calibrate_with_reasoing_paths == False:
            with open(f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_scores_path.pkl', 'wb') as f1: # TODO: Do not hard-code the names. It should include dataset name and some relevant model configuration
                pkl.dump(scores_path, f1)
            with open(f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_scores_a.pkl', 'wb') as f2:
                pkl.dump(scores_a, f2)
            with open(f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_scores_candidate.pkl', 'wb') as f3:
                pkl.dump(scores_candidate, f3)
        else:
            with open(f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_scores_path_rog.pkl', 'wb') as f1: # TODO: Do not hard-code the names. It should include dataset name and some relevant model configuration
                pkl.dump(scores_path, f1)
            with open(f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_scores_a_rog.pkl', 'wb') as f2:
                pkl.dump(scores_a, f2)
            with open(f'/home/bo/Dropbox/Projects/kg-llm-uq/calibrated_scores/{self.dataset}_scores_candidate_rog.pkl', 'wb') as f3:
                pkl.dump(scores_candidate, f3)

        return scores_path, scores_a, scores_candidate 
    
    def calculate_pairwise_score(self, s1, s2):
        if self.encoding_method == 'tfidf':
            return -cosine_similarity(self.model.transform([s1]), self.model.transform([s2]))
        if self.encoding_method == 'transformer':
            embedding_1= self.model.encode([s1, s2], convert_to_tensor=True)
            return -cosine_similarity(embedding_1[0].cpu().reshape(1, -1), embedding_1[1].cpu().reshape(1, -1)) # Could use torch function but wanted to keep it consistent

    def format_path(self, path):
        # Could be in utils instead of class function
        '''
            path is something like [(s, r, t), (s, r, t)]

            return:
                s -> r -> s -> r ... 
        '''  
        if len(path) == 0:
            return ""

        path_0 = "->".join(path[0])
        if len(path) > 1:
            final_path = path_0 + "->" + "->".join("->".join(triplet[1:]) for triplet in path[1:])
        else:
            final_path = path_0
        return final_path 

    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = graph_utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results
