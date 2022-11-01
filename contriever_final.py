# importing required libraries

import pandas as pd
import numpy as np
import json
import re
import os
import torch
from transformers import AutoTokenizer, AutoModel

# creating the class

class ContrieverCB:
    def __init__(self):
        self.embeddings = {}
    
    
    def clean(self, text: str) -> str:
        """
        Function to remove newline from text.
        :param text: input string
        :return: string without newline
        """
        new_text = re.sub('\n', '', text)
        return new_text
    
    
    def mean_pooling(self, token_embeddings, mask):
        """
        Function to be used after model is applied to tokenized text to generate embeddings.
        Used in the HuggingFace version.
        :param token_embeddings: output of model
        :param mask: attention mask of the tokens
        :return: tensors of the text
        """
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    
    def generate_embeddings(self, path_to_json: str, path_to_output: str) -> None:
        """
        Function takes input json filepath, generates numpy embeddings of the file and
        saves them at the given output filepath.
        :param path_to_json: input filepath
        :param path_to_output: output filepath
        :return: None
        """
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
        model = AutoModel.from_pretrained('facebook/contriever-msmarco')
        
        # open and read the input json file
        file = open(path_to_json)
        json_data = json.load(file)
        
        n = int(len(json_data)/100)
        embeddings_list = []
        
        # take 100 units at a time and process it
        for k in range(n):
            if k==n:
                start = k*100
                end = (list(json_data.keys())[-1])
            else:
                start = k*100
                end = k*100+99
                
            for i in range(start, end):
                text = json_data[str(i)]
                text = self.clean(text)
                tokenized_text = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                output = model(**tokenized_text)
                embeddings = self.mean_pooling(output[0], tokenized_text['attention_mask'])
                embeddings_np = embeddings.detach().numpy()
                embeddings_list.append(embeddings_np)
                
        # convert embeddings list to numpy array
        embeddings_array = np.array(embeddings_list)
        
        # reshape the numpy array
        x = embeddings_array.shape[0]
        y = embeddings_array.shape[2]
        embeddings_array.reshape((x,y))
        
        # save the embeddings in a numpy file
        # filename = path_to_json.split('\\')[-1].split('.')[0]
        # filepath = os.path.join(path_to_output, filename)
        filepath = path_to_output
        
        # saving the embeddings as a numpy file in the destination folder
        np.save(filepath, embeddings_array)
        
        # saving the embeddings into a dictionary
        self.embeddings[path_to_json] = embeddings_array
        
    
    def retrieve_topk(self, search_string: str, path_to_json: str, k: int):
        """
        Function takes json data as input and returns the topk relative to the data
        :param search_string: query to match and retrieve
        :param path_to_json: filepath of data to search
        :param k: number of embeddings to retrieve
        :return: top k units relative to the input file
        """
        
        # check if the embeddings are loaded in dictionary already
        if not (path_to_json in self.embeddings):
            
            # changing .json to .npy
            filename = os.path.splitext(path_to_json)[0]
            path_to_npy = filename + '.npy'
            
            # check if .npy exists and load into dictionary
            if os.path.exists(path_to_npy):
                self.embeddings[path_to_json] = np.load(path_to_npy)
            else:
                # .npy doesn't exist, so generate embeddings
                self.generate_embeddings(path_to_json, path_to_npy)
        
        # convert numpy embeddings to tensors
        embeddings = torch.from_numpy(self.embeddings[path_to_json])
        
        # convert query to tensor
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
        model = AutoModel.from_pretrained('facebook/contriever-msmarco')
        
        tokenized_query = tokenizer(search_string, padding=True, truncation=True, return_tensors='pt')
        output_query = model(**tokenized_query)
        embedded_query = self.mean_pooling(output_query[0], tokenized_query['attention_mask'])
        
        # creating a dictionary of scores
        scores_list = []
        for i in range(len(embeddings)):
            score = embeddings[i]@embedded_query[0]
            score_np = score.detach().numpy()[0]
            scores_list.append([i, score_np])
            
        scores_df = pd.DataFrame(scores_list, columns=['ID', 'Score'])
        
        # retrieving top k scores
        topk_scores = scores_df.nlargest(k, 'Score')
        
        # retrieving the text data corresponding to the top k indices
        topk_context = {}
        
        json_file = open(path_to_json)
        json_data = json.load(json_file)
        
        for row in topk_scores.iterrows():
            ind = int(row[1]['ID'])
            text = json_data[str(ind)]
            topk_context[ind] = text
        
        return topk_context







