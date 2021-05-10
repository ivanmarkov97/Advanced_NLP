from itertools import groupby

import torch

from dataset import LMDataset
from utils import choose_from_top


def infer(model: torch.nn.Module, dataset: LMDataset, input_str: str, max_len: int = 10) -> str:
	count = 0
	new_word = None
	result = input_str

	with torch.no_grad():
		model.eval()
		hidden, cell = model.init_hidden()

		for input_token in input_str.split():
			input_tensor = torch.tensor([dataset.text_to_ids([input_token])], dtype=torch.long)
			length_tensor = torch.tensor([1], dtype=torch.long)
			_, (hidden, cell) = model(input_tensor, length_tensor, (hidden, cell))

		while new_word != '<EOS>' and count < max_len:
			input_tensor = torch.tensor([dataset.text_to_ids(input_str.split())], dtype=torch.long)
			length_tensor = torch.tensor([len(input_str.split())], dtype=torch.long)
			predictions, (hidden, cell) = model(input_tensor, length_tensor, (hidden, cell))
			top_values, top_indexes = torch.topk(predictions, k=1, dim=1)
			top_values = top_values.numpy()[0]
			top_indexes = top_indexes.numpy()[0]
			top_index = choose_from_top(top_values, top_indexes)

			new_word = dataset.id2word[top_index]
			result = result + ' ' + new_word
			count += 1
		return ' '.join([word for word, _ in groupby(result.split())])
