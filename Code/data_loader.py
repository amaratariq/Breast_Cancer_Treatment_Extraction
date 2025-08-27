import torch
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