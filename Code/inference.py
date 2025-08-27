import pickle as pkl
import sys
import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import  classification_report
from models import QAModelGPT2, QAModelBioGPT

if __name__ == "__main__":
    '''
    arguments
    1: path to test dataframe - umls parsed and curated using data_curation file
    2: model_path - contians words "biogpt", "gpt2"
    3: save_dir  - where to save the model if it is to be finetuned, default path if none
    '''
    print('command line arguments', str(sys.argv))

    df_test = pd.reac_csv(sys.argv[1])
    model_path = sys.argv[2]
    save_dir = sys.argv[3]
    treatments_tokens = {"Surgery":'<SURGERY>',  
                    "Radiation": '<RADIATION>', 
                    "Chemotherapy":'<CHEMOTHERAPY>',
                    "Immunotherapy":'<TARGETTEDTHERAPY>',  
                    "Hormonetherapy":'<ENDOCRINE>'}
    treatments = ["Surgery",  "Radiation",  "Chemotherapy","Immunotherapy",  "Hormonetherapy"]
    treatments_cols = {"Surgery":'SURGERY',  
                        "Radiation": 'RADIATION', 
                        "Chemotherapy":'CHEMO',
                        "Immunotherapy":'IMMUNO',  
                        "Hormonetherapy":'HORMONE'}

    if "gpt2" in model_path:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel, PreTrainedModel
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        special_tokens_dict = {'additional_special_tokens': ['<SURGERY>',  '<RADIATION>', '<CHEMOTHERAPY>','<TARGETTEDTHERAPY>',  '<ENDOCRINE>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        special_tokens_dict = {'additional_special_tokens': ['<TRUE>', '<FALSE>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        qamodel = QAModelGPT2.from_pretrained(model_path)
        qamodel.resize_token_embeddings(len(tokenizer))
    else:
        from transformers import BioGptTokenizer, BioGptForCausalLM
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        tokenizer.pad_token = tokenizer.eos_token

        special_tokens_dict = {'additional_special_tokens': ['<SURGERY>',  '<RADIATION>', '<CHEMOTHERAPY>','<TARGETTEDTHERAPY>',  '<ENDOCRINE>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        special_tokens_dict = {'additional_special_tokens': ['<TRUE>', '<FALSE>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        qamodel = QAModelBioGPT.from_pretrained(model_path)
        qamodel.resize_token_embeddings(len(tokenizer))

    max_length = 512

    TRUE_TOKEN_ID = tokenizer("<TRUE>")["input_ids"][-1]
    FALSE_TOKEN_ID = tokenizer("<FALSE>")["input_ids"][-1]
    print("LABEL TOKENS:\t", TRUE_TOKEN_ID, FALSE_TOKEN_ID)
    sys.stdout.flush()


    def create_data_list(df):
        lst = []
        for i,j in df.iterrows():
            for tidx in range(len(treatments)):
                if df.at[i, treatments_cols[treatments[tidx]]]:
                    text = treatments_tokens[treatments[tidx]]+" "+df.at[i, "TEXT"]+" <TRUE>"
                else:
                    text = treatments_tokens[treatments[tidx]]+" "+df.at[i, "TEXT"]+" <FALSE>"
                lst.append(text)
        return lst
    data_list_test = create_data_list(df_test)

    print('datasets created')
    sys.stdout.flush()

    qamodel.eval()
    device = torch.device("cuda")
    qamodel = qamodel.to(device)

    cols = [t+"_prob" for t in treatments]+[t+"_gt" for t in treatments]
    df_test[cols] = None
    for i,j in df_test.iterrows():
        for tidx in range(len(treatments)):
            col_pred = treatments[tidx]+"_pred" #column to store the result
            col_gt = treatments[tidx]+"_gt" #column to store numeric gt label in 
            gt = df.at[i, treatments_cols[treatments[tidx]]]
            if gt is not None and gt:
                text = treatments_tokens[treatments[tidx]]+" "+df.at[i, "TEXT"]
                label = "<TRUE>"
                gt = 1
            else: #if label not provided - it will be set as false
                text = treatments_tokens[treatments[tidx]]+" "+df.at[i, "TEXT"]
                label = "<FALSE>"
                gt = 0
            label = torch.tensor(tokenizer(label)["input_ids"][-1]).unsqueeze(0).unsqueeze(0)
            try:
                inp = tokenizer(text, padding='max_length', max_length=max_length-1, truncation=True, return_tensors="pt") #keeping one token for label
            except:
                print(text)
            input_ids = torch.cat((inp["input_ids"],label), dim=1)
            attention_mask = torch.cat((inp["attention_mask"], torch.tensor([0]).unsqueeze(0)), dim=1) #ignore hr label
            attention_mask[input_ids==tokenizer.pad_token_id] = 0      

            out = qamodel(input_ids = x["input_ids"].to(device), attention_mask = x["attention_mask"].to(device))
            prob = torch.nn.Sigmoid()(out.logits[..., -1, -2]).detach().cpu().numpy()#pick the probability of <TRUE> , logits have been shifted inside

            df_test.at[i, col_pred] = prob.item()
            df_test.at[i, col_gt] = gt
    df_test.to_csv(save_dir+"test_set_w_predictions.csv", index=False)


'''
arguments
1: path to test dataframe - umls parsed and curated using data_curation file
2: model_path - contians words "biogpt", "gpt2"
3: save_dir  - where to save the model if it is to be finetuned, default path if none

python3 inference.py /path/to/curated/test/dataframe/  models/qamodel_biogpt_512_2013_2020_v2 ./

'''
