import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

import pickle as pkl
import sys
import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import  classification_report
from nltk import sent_tokenize, word_tokenize
from models import QAModelGPT2, QAModelBioGPT
from umls_parser import umls_parsing
from data_curation import curate_sample_per_patient, parse_umls_marked_notes
from data_loader import CustomDataset, DataCollatorForCustomDataset

print(torch.cuda.device_count())




if __name__ == "__main__":
    '''
    arguments
    1: data_dir - excel sheets for notes will be read from its subfolder "clinical_notes"; cui_lists as well
    2: do_train - true/false - if true, labels dataframe needed
    3: df_labels - needed when do_train=true; else None
    4: patients_split - pickle file with mrns dct for train test and val, needed is do_train is true
    5: model_path - contians words "biogpt", "gpt2"
    6: save_dir  - where to save the model if it is to be finetuned, default path if none
    '''
    print('command line arguments', str(sys.argv))

    data_dir = sys.argv[1]
    do_train = sys.srgv[2].lower()=="true"
    df_labels_path = sys.argv[3]
    patients_split = sys.argv[4]
    if do_train and (df_labels_path=="none" or  patients_split=="none"):
        print("labels and split files needed if model is to be trained/finetuned")
        return None
    model_path = sys.argv[5]
    save_dir = sys.argv[6]
    if save_dir == "none":
        save_dir = "model/"
        os.mkdir(save_dir)

    ## UMLS parsing and time interval based curation of notes and therapy labels (if registry data provided)
    data_files = list(os.listdir(data_dir+"clinical_notes/"))
    umls_marked_note_files_list = umls_parsing(data_files)
    df_labels = pd.read_csv(df_labels_path) #registry data
    df_labels["Clinic #"] =df_labels["Clinic #"].astype(str)
    
    if df_labels_path ! = "none": #if labels are provided
        df_labels = pd.read_csv(df_labels_path)
        df = parse_umls_marked_notes(umls_marked_note_files_list, df_labels)
    else:
        df = parse_umls_marked_notes(umls_marked_note_files_list)
    
    df.to_csv(data_dir+"umls_marked_time_curated_notes.csv", index=False)


    treatments_tokens = {"Surgery":'<SURGERY>',  
                    "Radiation": '<RADIATION>', 
                    "Chemotherapy":'<CHEMOTHERAPY>',
                    "Immunotherapy":'<TARGETTEDTHERAPY>',  
                    "Hormonetherapy":'<ENDOCRINE>'}
    treatments = ["Surgery",  "Radiation",  "Chemotherapy","Immunotherapy",  "Hormonetherapy"]
    treatment_thresholds = {"Surgery":0.31649184,  
                        "Radiation": 0.19697534, 
                        "Chemotherapy":0.13899182,
                        "Immunotherapy":0.15856697,  
                        "Hormonetherapy":0.23407674}

    treatments_cols = {"Surgery":'SURGERY',  
                        "Radiation": 'RADIATION', 
                        "Chemotherapy":'CHEMO',
                        "Immunotherapy":'IMMUNO',  
                        "Hormonetherapy":'HORMONE'}

    df = df.dropna(subset=["TEXT"])
    print(df.columns)

    if do_train:
        dct = pkl.load(open(patients_split, "rb"))
        mrn_train = dct["mrn_train"]
        mrn_test = dct["mrn_test"]
        mrn_val = dct["mrn_val"]

        df_train = df.loc[df.MRN.isin(mrn_train)]
        df_test = df.loc[df.MRN.isin(mrn_test)]
        df_val = df.loc[df.MRN.isin(mrn_val)]
        
    else:
        df_test = df.copy()

    print("data loaded")
    sys.stdout.flush()

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

    if do_train:
        data_list_train = create_data_list(df_train)
        data_list_val = create_data_list(df_val)
    data_list_test = create_data_list(df_test)

    if do_train:
        val_dataset = CustomDataset(tokenizer, data_list_val)
        train_dataset = CustomDataset(tokenizer, data_list_train)
        print(val_dataset.__len__(), train_dataset.__len__())
        
    
    test_dataset = CustomDataset(tokenizer, data_list_test)
    datacollator = DataCollatorForCustomDataset()
    print('datasets created')
    sys.stdout.flush()


    if do_train:

        from transformers import  Trainer, TrainingArguments
        from transformers import EarlyStoppingCallback, 


        training_args = TrainingArguments(
                    output_dir = save_dir,  # The output directory
                    overwrite_output_dir = True,  # overwrite the content of the output directory
                    num_train_epochs = 50,  # number of training epochs 5
                    per_device_train_batch_size = 16,  # batch size for training (ORIGINAL WAS 32)
                    per_device_eval_batch_size = 16,  # batch size for evaluation (ORIGINAL WAS 64)
                    gradient_accumulation_steps=4,

                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    logging_strategy='epoch',

                    save_total_limit=2,
                    warmup_steps = 100,  # number of warmup steps for learning rate scheduler 500
                    prediction_loss_only = False,

                    load_best_model_at_end = True,
                )

        trainer = Trainer(
            model = qamodel,
            args = training_args,
            data_collator = datacollator, #collate_fn,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
        )

        trainer.train()
        trainer.save_model()

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
    df_test.to_csv("test_set_w_predictions.csv", index=False)

    if df_labels_path != "none": #evaluate only if labels data was provided
        from roc_utils import *
        from sklearn.metrics import roc_auc_score
        for i in range(len(treatments)):
            print(treatments[i], end="\t")
            col_pred = treatments[tidx]+"_pred" #column to store the result
            col_gt = treatments[tidx]+"_gt" #column to store numeric gt label in
            probs_t =  df_test[col_pred].values
            labels_t = df_test[col_gt].values
            try:
                print(roc_auc_score(labels_t, probs_t))
            except:
                print("undefined")
            pred_labels = np.zeros((len(labels_t)))
            pred_labels[probs_t>treatment_thresholds[treatments[i]]]=1 #th] = 1

            print(classification_report(labels_t, pred_labels))
            sys.stdout.flush()


    
'''
arguments
1: data_dir - excel sheets for notes will be read from here; cui_lists as well
2: do_train - true/false - if true, labels dataframe needed
3: df_labels_path- needed when do_train=true; else None
4: patients_split - pickle file with mrns dct for train test and val, needed is do_train is true
5: model_path - contians words "biogpt", "gpt2"
6: save_dir

python3 breast_cancer_treatment_prediction_pipeline.py data_dir/ true registry_data.csv patients_split.pkl  models/qamodel_biogpt_512_2013_2020_v2 ./

'''
