import pandas as pd
import os
import sys
import numpy as np
import re
from datetime import timedelta, datetime
from tqdm.auto import tqdm

def curate_sample_per_patient(df_notes_mrn, df_labels=None): #notes of one mrn
    '''
    df_notes_mrn: csv of all notes (umls parsed) for one patient mrn
    df_labels: cancer registry data; if provided, groundtruth labels will be curated in terms of time intervals for each therapy
    return list of start and ed dates of interval and label for each treatment within in each interval

    '''

    mrn = df_notes_mrn.PATIENT_CLINIC_NUMBER.values[0] #the whole dataframe should have the same MRN
    df_notes_mrn["NOTE_DATE"] = pd.to_datetime(df_notes_mrn.CLINICAL_DOCUMENT_ORIGINAL_DTM).dt.date
    df_notes_mrn = df_notes_mrn.sort_values(by=["NOTE_DATE"])
    st_date = df_notes_mrn.NOTE_DATE.values[0]
    ed_date = df_notes_mrn.NOTE_DATE.values[-1]

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
            radiation_ed_dt = datetime(2050, 12, 31).date()
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
            chemo_ed_dt = datetime(2050, 12, 31).date()
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
            immuno_ed_dt = datetime(2050, 12, 31).date()

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
    '''
    data_files: list of csv files containing notes with UMLS extraction

    call curate_sample_per_patient() after collecting all notes of one patient

    df_labels: registry data, should have the following columns

    'Surg, Date Most Def Tx',
    'Surg, Date of Tx, Hist',
    'Rad, Summ Date Start',
    'Rad, Summ Date End',
    'Chemo, Summ Date Start',
    'Chemo, Summ Date End',
    'Hormone, Summ Date Started',
    'Immuno, Summ Date Start'

    return dataframe; one row corresponds to one time interval of one patient along with labels (Yes/No) for every therapy 
    
    '''
    idx = 0
    df = pd.read_csv(umls_marked_file_paths) #assume PATIENT_CLINIC_NUMBER column is available - can be merged with "Clinic #" column of df_labels
    print(len(df), end = "\t")
    if df_labels is not None:
        df = df.loc[df.PATIENT_CLINIC_NUMBER.isin(df_labels["Clinic #"].values)]
        print(len(df))

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
    return df_curated