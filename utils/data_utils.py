import json
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle as pkl
from tqdm import tqdm
from utils.graph_utils import build_graph
from datasets import load_dataset, DatasetDict, concatenate_datasets

def split_trainset(train_set, split_rate=0.3): # TODO: Increase in real setting
    """
        Splitting training set into training and caliberation. Default to 30% caliberation set.
        return: training set (30% of original data), and caliberation set(70% of original data)
    """
    split_datasets = train_set.train_test_split(test_size=split_rate)

    return split_datasets['train'], split_datasets['test']

def pretrain_tfidf():
    file_path = "/home/bo/Dropbox/Projects/kg-llm-uq/tfidf_vectorizer.pkl"
    if (os.path.isfile(file_path)):
        with open(file_path, 'rb') as f:
            return pkl.load(f) 
    else: 
        dataset = load_dataset(dataset)
        dataset = DatasetDict({"train": concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])})
    
        sentences = []
        for data in tqdm(dataset['train']):
            graph = build_graph(data['graph']) 
            for (u, v) in graph.edges:
                r = graph[u][v]['relation']
                sentences.append(u + " " + r + " " + v)
        vectorizer = TfidfVectorizer()
        vectorizer.fit(sentences)
        with open(file_path, 'wb') as f:
            pkl.dump(vectorizer, f)

    return vectorizer

def read_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt_template = f"""{f.read()}"""
    return prompt_template

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_multiple_jsonl(file_path_list):
    data = []
    for path in file_path_list:
        data.extend(load_jsonl(path))
    return data

def list_to_string(l: list) -> str:
    prompt = '"{}"'
    return ', '.join([prompt.format(i) for i in l])

def rule_to_string(rule: list, sep_token = "<SEP>", bop = "<PATH>", eop = "</PATH>") -> str:
    if len(rule) == 1:
        rule_string = rule[0]
    else:
        rule_string = sep_token.join(rule)
    return bop + rule_string + eop

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

class InstructFormater(object):
    def __init__(self, prompt_path):
        '''
        _summary_

        Args:
            prompt_template (_type_): 
            instruct_template (_type_): _description_
        '''
        self.prompt_template = read_prompt(prompt_path)

    def format(self, instruction, message):
        return self.prompt_template.format(instruction=instruction, input=message)
    
import random
import math

class NoDuplicatesDataLoader:
    def __init__(self, train_examples, batch_size):
        """
        A special data loader to be used with MultipleNegativesRankingLoss.
        The data loader ensures that there are no duplicate sentences within the same batch
        """
        self.batch_size = batch_size
        self.data_pointer = 0
        self.collate_fn = None
        self.train_examples = train_examples
        random.shuffle(self.train_examples)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch = []
            texts_in_batch = set()

            while len(batch) < self.batch_size:
                example = self.train_examples[self.data_pointer]

                valid_example = True
                for text in example.texts:
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch.append(example)
                    for text in example.texts:
                        texts_in_batch.add(text.strip().lower())

                self.data_pointer += 1
                if self.data_pointer >= len(self.train_examples):
                    self.data_pointer = 0
                    random.shuffle(self.train_examples)

            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        return math.floor(len(self.train_examples) / self.batch_size)