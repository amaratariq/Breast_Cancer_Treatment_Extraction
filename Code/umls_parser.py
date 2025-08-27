import pandas as pd
import os
import sys
import numpy as np
from quickumls import QuickUMLS
from nltk import sent_tokenize, word_tokenize
from striprtf.striprtf import rtf_to_text 



def umls_parsing(data_files, cui_therapy, cui_meds):
    '''
    data_files: list of excel files containing notes; one file may have multiple sheet
    returns list of paths to files with UMLS parsed text corresponding to each note
    '''
    ## read lists of relevant concepts for therapies and medication for breast cancer in UMLS vocabulary
    cui_therapy = list(pd.read_excel("../Data/Copy of thearpy_cui 2_AK comments.xlsx")["Unnamed: 1"][1:])
    cui_meds = list(pd.read_excel("../Data/Copy of thearpy_med_cui_AK comments.xlsx")["Unnamed: 1"][1:])

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