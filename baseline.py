import torch
from torch.utils.data import Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset, concatenate_datasets

RANDOM_SEED = 42

class CombinedDataset(Dataset):
    def __init__(d):
        self.data = d

    def _len_(self):
        return len(self.data)

    def _getitem_(self, index):
        example = self.data[index]
        question = example["question"]
        context = example["context"]

        # Tokenize the question and context
        inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)

        # Get the predicted answer from the model
        outputs = model(**inputs)
        answer_length = len(tokenizer.tokenize(outputs.start_logits.argmax() + outputs.end_logits.argmax()))

        return answer_length, len(inputs["input_ids"].squeeze())

def load_data(src_prefix="squad", known_prefix="hotpot_qa", unknown_prefix="newsqa"):
    p_src = load_dataset(src_prefix, split='train', streaming=True)
    dataset1 = p_src.shuffle(seed=RANDOM_SEED, buffer_size=1600)
    d1 = parse(dataset1)

    q_known = load_dataset(known_prefix, "distractor", split='train', streaming=True)
    dataset2 = q_known.shuffle(seed=RANDOM_SEED, buffer_size=400)
    d2 = parse(dataset2)

    # q_unknown = load_dataset(unknown_prefix, streaming=True)
    # dataset3 = q_unknown.shuffle(seed=RANDOM_SEED, buffer_size=4000)

    dataset = d1 + d2

    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    data_loader = DataLoader(CombinedDataset(dataset), batch_size=32, shuffle=True)

    return data_loader

parse = lambda dataset: [{'question': example["question"], 'context': example["context"]} for example in dataset]

