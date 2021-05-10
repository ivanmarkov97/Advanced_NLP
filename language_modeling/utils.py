import os
import re
import random

from typing import List, Dict, Any

import torch
import numpy as np


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		if torch.cuda.is_available():
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = True


def process(text: str) -> str:
	text = text.lower()
	text = re.sub('[-",:!\t]', '', text)
	text = re.sub('\n', ' ', text)
	text = text.strip()
	return text


def post_process(text: str, special_chars: str) -> str:
	text = re.sub(f'[{special_chars}]', '', text)
	return text


def collate_fn(items: List[Dict[str, torch.Tensor]], pad_value: int):
	item_keys = items[0].keys()
	items_max_size = {item: 0 for item in item_keys}
	total_collection = {key: [] for key in item_keys}

	for item in items:
		for key in item:
			if items_max_size[key] < item[key].shape[1]:
				items_max_size[key] = item[key].shape[1]

	for item in items:
		for key in item_keys:
			pad_size = items_max_size[key] - item[key].shape[1]
			pad_tensor = torch.zeros((1, pad_size), dtype=item[key].dtype)
			pad_tensor.fill_(pad_value)
			item[key] = torch.cat((item[key], pad_tensor), dim=1)
			total_collection[key].append(item[key])

	for key in total_collection:
		total_collection[key] = torch.cat(total_collection[key], dim=0)

	return total_collection


def choose_from_top(p: np.ndarray, values: np.ndarray) -> Any:
	p = np.abs(p)
	p = np.exp(p) / np.sum(np.exp(p))
	value = np.random.choice(values, p=p)
	return value
