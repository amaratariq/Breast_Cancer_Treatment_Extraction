Pipeline includes all processing steps. You can use or remove any steps based on your own preprocessing.

- Marking of clincial notes with sentences mentioing relevant medication and treatment concepts
- Parsing of cancer registry data for assigning labels for evaluation or finetuning or training
- Model training
- Inference

Input arguments are listed below.

- data_dir - excel sheets for notes will be read from subfolder clinical_notes; cui_lists are placed in this folder
- do_train - true/false - if true, labels dataframe needed
- df_labels_path- needed when do_train=true; else None
- patients_split - pickle file with dictionary object with three keys; mrn_train, mrn_test, mrn_val, cntianing lists of MRNs of patients in each split.
- model_path - contians words "biogpt", "gpt2" (will be provided upon reasonable request)
- save_dir

Sample command is as follows.

```python3 pipeline.py ../Data/ true registry_data.csv patients_split.pkl  models/qamodel_biogpt_512_2013_2020_v2 ./```
