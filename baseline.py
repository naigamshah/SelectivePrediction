import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset

def load_data(src_prefix="squad", known_prefix="hotpot", unknown_prefix="news"):

	p_src = load_dataset(src_prefix)
	q_known = load_dataset(known_prefix)
	q_unknown = load_dataset(unknown_prefix)

	combined_dataset = dataset1[:1600] + dataset2[:400] + dataset3[:4000]

	model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    class CombinedDataset(Dataset):
        def _len_(self):
            return len(combined_dataset)

        def _getitem_(self, index):
            example = combined_dataset[index]
            question = example["question"]
            context = example["context"]

            # Tokenize the question and context
            inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)

            # Get the predicted answer from the model
            outputs = model(**inputs)
            answer_length = len(tokenizer.tokenize(outputs.start_logits.argmax() + outputs.end_logits.argmax()))

            return answer_length, len(inputs["input_ids"].squeeze())

   data_loader = DataLoader(CombinedDataset(), batch_size=32, shuffle=True)

    return data_loader