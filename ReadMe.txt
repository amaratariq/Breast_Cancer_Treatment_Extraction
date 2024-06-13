Pipeline includes all processing steps. You can use or remove any steps based on your own preprocessing.

** 1- Marking of clincial notes with sentences mentioing relevant medication and treatment concepts
** 2- Parsing of cacner registry data for assigning labels for evaluation or finetuning or training
** 3- Model training
** 4- Inference

Input arguments are listed below.

1: data_dir - excel sheets for notes will be read from subfolder clinical_notes; cui_lists are placed in this folder
2: do_train - true/false - if true, labels dataframe needed
3: df_labels_path- needed when do_train=true; else None
4: patients_split - pickle file with dictionary object with three keys; mrn_train, mrn_test, mrn_val, cntianing lists of MRNs of patients in each split.
5: model_path - contians words "biogpt", "gpt2" (will be provided upon reasonable request)
6: save_dir

Sample command is as follows.

```python3 breast_cancer_treatment_prediction_pipeline.py data_dir/ true registry_data.csv patients_split.pkl  models/qamodel_biogpt_512_2013_2020_v2 ./'''