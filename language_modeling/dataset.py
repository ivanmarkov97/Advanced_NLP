from typing import List, Dict

import torch

from torch.utils.data import Dataset


class ProcessText(object):
	def __init__(self, sent_split_chars, pre_processing_func, post_processing_func=None) -> None:

		self.sent_split_chars = sent_split_chars

		if pre_processing_func is not None:
			self.pre_processing_func = pre_processing_func
		else:
			self.pre_processing_func = lambda v: v

		if post_processing_func is not None:
			self.post_processing_func = post_processing_func
		else:
			self.post_processing_func = lambda v: v

	def transform(self, path: str) -> List:
		all_text = []
		curr_sentence = []
		with open(path, 'r', encoding='utf-8') as f:
			for text in f.readlines():
				text = self.pre_processing_func(text)

				if text == '':
					continue

				curr_sentence.append(text)

				if text[-1] in self.sent_split_chars:
					sentence = ' '.join(curr_sentence)
					sentence = self.post_processing_func(sentence)
					tokens = sentence.split()
					if tokens:
						all_text.append(tokens)
						curr_sentence = []
		return all_text


class LMDataset(Dataset):

	def __init__(self, path: str, process: ProcessText = None) -> None:

		self.word2id = {'<EOS>': 0, '<pad>': 1, '<UNK>': 2}

		if process is not None:
			self.texts = process.transform(path)
		else:
			print('Text processing is None. Gonna read text')
			self.texts = []

			with open(path, 'r', encoding='utf-8') as f:
				for line in f.readlines():
					self.texts.append(line.split())

		for text in self.texts:
			for word in text:
				if word not in self.word2id:
					self.word2id[word] = len(self.word2id)

		self.id2word = dict([(index, word) for word, index in self.word2id.items()])

	def text_to_ids(self, text: List[str]) -> List[int]:
		return [self.word2id[word] for word in text]

	def __getitem__(self, index) -> Dict[str, torch.Tensor]:
		input_text = self.texts[index]
		output_text = input_text[1:]
		output_text.append('<EOS>')

		text_len = torch.tensor([[len(input_text)]], dtype=torch.long)
		input_text_ids = torch.tensor([self.text_to_ids(input_text)], dtype=torch.long)
		output_text_ids = torch.tensor([self.text_to_ids(output_text)], dtype=torch.long)

		return {
			'input': input_text_ids,
			'output': output_text_ids,
			'length': text_len
		}

	def __len__(self) -> int:
		return len(self.texts)
