import os

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

import pickle as pkl
import sys
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, GPT2LMHeadModel,GPT2Tokenizer
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import  classification_report

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions
)
from typing import Optional, Tuple, Union

from quickumls import QuickUMLS
import numpy as np
from nltk import sent_tokenize, word_tokenize
from striprtf.striprtf import rtf_to_text 

print(torch.cuda.device_count())


def umls_parsing(data_files, cui_therapy, cui_meds):
    matcher = QuickUMLS('UMLS', threshold=0.8)
    print("files to be parsed:\n", data_files)

    save_paths = []
    for fidx in range(len(data_files)):
        fname = data_files[fidx]
        print("reading:", fname)
        xls = pd.ExcelFile(fname)
        print("sheets:", xls.sheet_names)
        sys.stdout.flush()
        for sh in xls.sheet_names:
            
            save_name = fname+"_"+sh+".csv"
            print("will be saved as:", save_name)
            save_paths.append(save_name)

            df = pd.read_excel(xls, sh)
            st = 0
            sys.stdout.flush()
            for idx in range(st, len(df)):#i,j in df.iterrows():
                i = df.index[idx]
                concept_ids = ""
                concepts = ""
                sents_sel = ""
                flag = False 
                try:
                    txt = df.at[i, 'CLINICAL_DOCUMENT_TEXT']
                    txt = txt.replace("\n", " ")
                    txt = txt.replace("\t", " ")
                    txt = " ".join(txt.split())
                    txt = rtf_to_text(txt)
                    sents = sent_tokenize(txt)
                    flag = True
                except:
                    print("ISSUE with note")
                    flag = False
                if flag:
                    for s in sents:
                        try:
                            if len(word_tokenize(s))>3 and len(word_tokenize(s))<100:
                                out = matcher.match(s, best_match=True, ignore_syntax=False)
                                if len(out)>0:
                                    cuis = []
                                    for o in out:
                                        cuis = cuis + [oo["cui"] for oo in o if oo["cui"] in cui_meds or oo["cui"] in cui_therapy]
                                    cuis = list(np.unique(cuis))
                                    if len(cuis)>0:
                                        concept_ids = concept_ids+"|"+"|".join(cuis)
                                        cuis = []
                                        for o in out:
                                            cuis = cuis + [oo["ngram"] for oo in o if oo["cui"] in cui_meds or oo["cui"] in cui_therapy]
                                        cuis = list(np.unique(cuis))
                                        concepts = concepts+"|"+"|".join(cuis)
                                        sents_sel = sents_sel+"|"+s
                        except:
                            print("PROBLEM sentences")
                            sys.stdout.flush()
                df.at[i, "CUI"] = concept_ids
                df.at[i, "concepts"] = concepts
                df.at[i, "sentences"] = sents_sel
                for c in df.columns:
                    df.at[i, c] = df.at[i, c]
                if i%10000==0 and i > st:
                    df.to_csv(save_name, index=False)
                    print("saving", i, "of", len(df))
                    sys.stdout.flush()
            df.to_csv(save_name, index=False)
    return save_paths

def curate_sample_per_patient(df_notes_mrn, df_labels=None): #notes of one mrn
    # global df #notes data
    # global dfout #registry data
    # temp = df.loc[df.PATIENT_CLINIC_NUMBER==mrn]

    df_notes_mrn["NOTE_DATE"] = pd.to_datetime(df_notes_mrn.CLINICAL_DOCUMENT_ORIGINAL_DTM).dt.date
    df_notes_mrn = df_notes_mrn.sort_values(by=["NOTE_DATE"])
    st_date = temp.NOTE_DATE.values[0]
    ed_date = temp.NOTE_DATE.values[-1]

    st_dates = []
    ed_dates = []
    inputs = []
    surgery =[]
    radiation = []
    chemo = []
    hormone = []
    immuno = []
    if df_labels is not None:
        out = df_labels.loc[df_labels["Clinic #"]==str(int(mrn))]
        out = out.drop_duplicates(subset=["Clinic #"], keep="first")
        ## surgery
        if type(out['Surg, Date Most Def Tx'].values[0]) is str and  (out['Surg, Date Most Def Tx'].values[0].startswith("20")):#   (out['Surg, Date Most Def Tx'].values[0] != "88/88/8888") and (out['Surg, Date Most Def Tx'].values[0] != "00/00/0000"):
            dt = pd.to_datetime(out['Surg, Date Most Def Tx']).dt.date.values[0]  #will fail for NaN
            surgery_st_dt = dt-timedelta(days = 180)
            surgery_ed_dt = dt+timedelta(days = 180)
        elif type(out['Surg, Date of Tx, Hist'].values[0]) is str  and  (out['Surg, Date of Tx, Hist'].values[0].startswith("20")): #(out['Surg, Date of Tx, Hist'].values[0] != "88/88/8888") and (out['Surg, Date of Tx, Hist'].values[0] != "00/00/0000"): # check hist if Def unavailbale
            dt = pd.to_datetime(out['Surg, Date of Tx, Hist']).dt.date.values[0]  #will fail for NaN
            surgery_st_dt = dt-timedelta(days = 180)
            surgery_ed_dt = dt+timedelta(days = 180)
        else:
            surgery_st_dt = datetime(2050, 12, 31).date()
            surgery_ed_dt = datetime(2050, 12, 31).date()
    
        ## radiation'Rad, Summ Date Start', 'Rad, Summ Date End',
        radiation_st_dt = None
        radiation_ed_dt = None
        col = 'Rad, Summ Date Start'
        # print(out[col].values[0])
        if type(out[col].values[0]) is str and (out[col].values[0].startswith("20")):# (out[col].values[0]!= "88/88/8888") and (out[col].values[0] != "00/00/0000"):  #Def
            dt = pd.to_datetime(out['Rad, Summ Date Start']).dt.date.values[0]  #will fail for NaN
            radiation_st_dt = dt-timedelta(days = 180)
        col ='Rad, Summ Date End'
        if type(out[col].values[0]) is str and (out[col].values[0].startswith("20")):#(out[col].values[0]!= "88/88/8888") and (out[col].values[0] != "00/00/0000"): #Def
            dt = pd.to_datetime(out['Rad, Summ Date End']).dt.date.values[0]  #will fail for NaN
            radiation_st_dt = dt+timedelta(days = 180)
        if radiation_st_dt is None and radiation_ed_dt is None:
            radiation_st_dt = datetime(2050, 12, 31).date()
            radiation_st_ed = datetime(2050, 12, 31).date()
        if radiation_st_dt is None and radiation_ed_dt is not None:
            radiation_st_dt = radiation_ed_dt-timedelta(days = 180)
        if radiation_st_dt is not None and radiation_ed_dt is None:
            radiation_ed_dt = radiation_st_dt+timedelta(days = 180)

        ## chemo 'Chemo, Summ Date Start', 'Chemo, Summ Date End',,
        chemo_st_dt = None
        chemo_ed_dt = None
        col = 'Chemo, Summ Date Start'
        if type(out[col].values[0]) is str and (out[col].values[0].startswith("20")):#(out[col].values[0]!= "88/88/8888") and (out[col].values[0] != "00/00/0000"):
            dt = pd.to_datetime(out['Chemo, Summ Date Start']).dt.date.values[0]  #will fail for NaN
            chemo_st_dt = dt-timedelta(days = 180)
        col = 'Chemo, Summ Date End'
        if type(out[col].values[0]) is str and (out[col].values[0].startswith("20")):#(out[col].values[0]!= "88/88/8888") and (out[col].values[0] != "00/00/0000"):  
            dt = pd.to_datetime(out['Chemo, Summ Date End']).dt.date.values[0]  #will fail for NaN
            chemo_st_dt = dt+timedelta(days = 180)
        if chemo_st_dt is None and chemo_ed_dt is None:
            chemo_st_dt = datetime(2050, 12, 31).date()
            chemo_st_ed = datetime(2050, 12, 31).date()
        if chemo_st_dt is None and chemo_ed_dt is not None:
            chemo_st_dt = chemo_ed_dt-timedelta(days = 180)
        if chemo_st_dt is not None and chemo_ed_dt is None:
            chemo_ed_dt = chemo_st_dt+timedelta(days = 180)  

        ## immunotherapy 'Immuno, Summ Date Start'
        immuno_st_dt = None
        immuno_ed_dt = None
        col = 'Immuno, Summ Date Start'
        if type(out[col].values[0]) is str and (out[col].values[0].startswith("20")):#(out[col].values[0]!= "88/88/8888") and (out[col].values[0] != "00/00/0000"): 
            dt = pd.to_datetime(out['Immuno, Summ Date Start']).dt.date.values[0]  #will fail for NaN
            immuno_st_dt = dt-timedelta(days = 180)
            immuno_ed_dt = dt+timedelta(days = 180)
        if immuno_st_dt is None:
            immuno_st_dt = datetime(2050, 12, 31).date()
            immuno_st_ed = datetime(2050, 12, 31).date()

        ## hormonetherapy 'Hormone, Summ Date Start'
        hormone_st_dt = None
        hormone_ed_dt = None
        col = 'Hormone, Summ Date Started'
        # print(out[col].values[0])
        if type(out[col].values[0]) is str and(out[col].values[0].startswith("20")):#(out[col].values[0]!= "88/88/8888") and (out[col].values[0] != "00/00/0000"):  
            dt = pd.to_datetime(out['Hormone, Summ Date Started']).dt.date.values[0]  #will fail for NaN
            hormone_st_dt = dt-timedelta(days = 180)
            hormone_ed_dt = dt+timedelta(days = 180)
        if hormone_st_dt is None:
            hormone_st_dt = datetime(2050, 12, 31).date()
            hormone_ed_dt = datetime(2050, 12, 31).date()
        
        # print(surgery_st_dt, surgery_ed_dt, radiation_st_dt, radiation_ed_dt, chemo_st_dt, chemo_ed_dt, immuno_st_dt, immuno_ed_dt, hormone_st_dt, hormone_ed_dt)
    
    st_interval = st_date
    while st_interval<ed_date:
        ed_interval = st_interval+timedelta(days = 6*30)
        # print(st_interval, ed_interval)
        temp_interval = df_notes_mrn.loc[(df_notes_mrn.NOTE_DATE>=st_interval) & (df_notes_mrn.NOTE_DATE<=ed_interval)]
        text = "|".join(temp_interval.dropna(subset=["sentences"]).sentences.values)
        
        if df_labels is not None:
            surgery_label = (st_interval>=surgery_st_dt and st_interval<=surgery_ed_dt) or (ed_interval>=surgery_st_dt and ed_interval<=surgery_ed_dt)
            radiation_label = (st_interval>=radiation_st_dt and st_interval<=radiation_ed_dt) or (ed_interval>=radiation_st_dt and ed_interval<=radiation_ed_dt)
            chemo_label = (st_interval>=chemo_st_dt and st_interval<=chemo_ed_dt) or (ed_interval>=chemo_st_dt and ed_interval<=chemo_ed_dt)
            immuno_label = (st_interval>=immuno_st_dt and st_interval<=immuno_ed_dt) or (ed_interval>=immuno_st_dt and ed_interval<=immuno_ed_dt)
            hormone_label = (st_interval>=hormone_st_dt and st_interval<=hormone_ed_dt) or (ed_interval>=hormone_st_dt and ed_interval<=hormone_ed_dt)
        else:
            surgery_label = None
            radiation_label = None
            chemo_label = None
            immuno_label = None
            hormone_label = None
        inputs.append(text)
        surgery.append(surgery_label)
        radiation.append(radiation_label)
        chemo.append(chemo_label)
        hormone.append(hormone_label)
        immuno.append(immuno_label)

        st_dates.append(st_interval)
        ed_dates.append(ed_interval)


        st_interval = ed_interval

    return st_dates, ed_dates, inputs, surgery, radiation, chemo, immuno, hormone

def parse_umls_marked_notes(umls_marked_file_paths, df_labels=None):
    idx = 0
    df = pd.read_csv(umls_marked_file_paths) #assume PATIENT_CLINIC_NUMBER column is available - can be merged with "Clinic #" column of df_labels
    print(len(df), end = "\t")
    if df_labels is not None:
        df = df.loc[df.PATIENT_CLINIC_NUMBER.isin(df_labels["Clinic #"].values)]
        print(len(df))
    print()

    for f in umls_marked_file_paths[1:]:
        temp = pd.read_csv(f)
        print(len(temp), end = "\t")
        if df_labels is not None:
            temp = temp.loc[temp.PATIENT_CLINIC_NUMBER.isin(df_labels["Clinic #"].values)]
        print(len(temp))
        df = pd.concat([df, temp], ignore_index=True)

    if df_labels is not None:
        cols_sel_dates = ['Surg, Date Most Def Tx',
        'Surg, Date of Tx, Hist',
        'Rad, Summ Date Start',
        'Rad, Summ Date End',
        'Chemo, Summ Date Start',
        'Chemo, Summ Date End',
        'Hormone, Summ Date Started',
        'Immuno, Summ Date Start']
        for c in cols_sel_dates:
            df_labels[c] = pd.to_datetime(df_labels[c], errors = "ignore")

    mrns_sel = df.PATIENT_CLINIC_NUMBER.unique()
    df_curated = pd.DataFrame(columns = ["MRN", "ST_DATE", "ED_DATE", "TEXT", "SURGERY", "RADIATION", "CHEMO", "IMMUNO", "HORMONE"])
    k = 0
    for m in tqdm(mrns_sel):
        notes_m = df.loc[df.PATIENT_CLINIC_NUMBER==m]
        st_dates, ed_dates, inputs, surgery, radiation, chemo, immuno, hormone = curate_sample_per_patient(notes_m, df_labels)
        for i in range(len(st_dates)):
            df_curated.at[k, "MRN"] = m
            df_curated.at[k, "ST_DATE"] = st_dates[i]
            df_curated.at[k, "ED_DATE"] = ed_dates[i]
            df_curated.at[k, "TEXT"] = inputs[i]
            df_curated.at[k, "SURGERY"] = surgery[i]
            df_curated.at[k, "RADIATION"] = radiation[i]
            df_curated.at[k, "CHEMO"] = chemo[i]
            df_curated.at[k, "IMMUNO"] = immuno[i]
            df_curated.at[k, "HORMONE"] = hormone[i]
            k+=1
    #     # print(len(df_curated))
    #     if k%500 == 0 and k>0:
    #         print(k)
    #         df_curated.to_csv("curated_clean_data_with_labels_2013_2020.csv", index=False)
    # df_curated.to_csv("curated_clean_data_with_labels_2013_2020.csv", index=False)
    return df_curated

from transformers import GPT2LMHeadModel
class QAModelGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
       
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True, #None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)


        lm_logits = lm_logits[..., :-1, :] #shifitng
        out = lm_logits[...,-1, -2:] #TRUE and FALSE probs only - after shifitng
        loss = None
        if labels is not None:
            labels = labels[..., -1].clone()
            labels[labels == TRUE_TOKEN_ID] = 0 # TRUE token is 50262/263 , 2nd to the last column of lm_logits, 0th column of out
            labels[labels != 0] = 1
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            out = out.contiguous()
            labels = labels.contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(out.view(-1, out.size(-1)), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )





from transformers import BioGptTokenizer, BioGptForCausalLM
class QAModelBioGPT(BioGptForCausalLM):
    def __init__(self, config):
        super().__init__(config)
       

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(return_dict)
        outputs = self.biogpt(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(outputs)
        sequence_output = outputs[0]
        prediction_scores = self.output_projection(sequence_output)

        prediction_scores = prediction_scores[..., :-1, :] #shifitng
        out = prediction_scores[...,-1, -2:] #TRUE and FALSE probs only - after shifitng

        lm_loss = None
        if labels is not None:

            labels = labels[..., -1].clone()
            labels[labels == TRUE_TOKEN_ID] = 0 # TRUE token is  2nd to the last column of lm_logits, 0th column of out
            labels[labels != 0] = 1
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            out = out.contiguous()
            labels = labels.contiguous()
            loss_fct = CrossEntropyLoss()#weight=torch.Tensor([POS_WEIGHT, NEG_WEIGHT]).to(prediction_scores.device))
            lm_loss = loss_fct(out.view(-1, out.size(-1)), labels.view(-1))


        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

from torch.utils.data import Dataset
class CustomDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer, data):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data = data
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        text = self.data[i]

        if text.endswith("<TRUE>"):
            label = "<TRUE>"
            text = text.replace("<TRUE>", "")
        else:
            label = "<FALSE>"
            text = text.replace("<FALSE>", "")


        label = torch.tensor(self.tokenizer(label)["input_ids"][-1]).unsqueeze(0).unsqueeze(0)
        # label = torch.tensor(self.tokenizer(label)["input_ids"]).unsqueeze(0)
        try:
            inp = self.tokenizer(text, padding='max_length', max_length=max_length-1, truncation=True, return_tensors="pt") #keeping one token for label
        except:
            print(text)
        input_ids = torch.cat((inp["input_ids"],label), dim=1)

        attention_mask = torch.cat((inp["attention_mask"], torch.tensor([0]).unsqueeze(0)), dim=1) #ignore hr label
        attention_mask[input_ids==self.tokenizer.pad_token_id] = 0
                
        label_ids = input_ids.clone()
        label_ids[0, :-1]= -100 #ignore everything but the label token

        return dict(input_ids=input_ids, labels=label_ids, attention_mask=attention_mask)



class DataCollatorForCustomDataset(object):
    """Collate examples for custom dataset"""

    def __call__(self, examples_in):
        # print(examples_in)
        examples = [e["input_ids"] for e in examples_in]
        #we know everything is set for tokenize routput, padding and all
        inputs = examples[0].new_full([len(examples), examples[0].shape[1]], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            inputs[i, :] = example


        examples = [e["labels"] for e in examples_in]
        labels = examples[0].new_full([len(examples), examples[0].shape[1]], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            labels[i, :] = example

        examples = [e["attention_mask"] for e in examples_in]
        attention_mask = examples[0].new_full([len(examples), examples[0].shape[1]], tokenizer.pad_token_id)

        for i, example in enumerate(examples):
            attention_mask[i, :] = example


        return {"input_ids": inputs, "labels": labels, "attention_mask": attention_mask}


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

    ## UMLS parsing of notes
    cui_therapy = list(pd.read_excel(data_dir+"Copy of thearpy_cui 2_AK comments.xlsx")["Unnamed: 1"][1:])
    cui_meds = list(pd.read_excel(data_dir+"Copy of thearpy_med_cui_AK comments.xlsx")["Unnamed: 1"][1:])
    data_files = list(os.listdir(data_dir+"clinical_notes/"))
    umls_marked_note_files_list = umls_parsing(data_files, cui_therapy, cui_meds)

    ## notes parsing (time interval) before extraction

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
