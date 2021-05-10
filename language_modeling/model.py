from typing import Dict, Union, Tuple

import torch

from torch import nn
from torch.nn import functional as F


class RNN(nn.Module):

	def __init__(self, config: Dict[str, Union[int, float]]) -> None:
		super().__init__()
		self.config = config
		self.embedding = nn.Embedding(config['vocab_size'], config['emb_size'], padding_idx=config['pad_idx'])

		self.rnn = nn.LSTM(config['emb_size'], config['hid_size'], config['n_layers'],
						   dropout=config['dropout'] if config['n_layers'] > 1 else 0, batch_first=True)

		self.output = nn.Linear(config['hid_size'], config['vocab_size'])

	def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
		hidden = torch.zeros((self.config['n_layers'], self.config['bs'], self.config['hid_size']))
		cell = torch.zeros((self.config['n_layers'], self.config['bs'], self.config['hid_size']))
		nn.init.uniform_(hidden, a=-0.3, b=0.3)
		nn.init.uniform_(cell, a=-0.3, b=0.3)
		return hidden, cell

	def forward(self, text, length, hidden) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
		embedded = self.embedding(text)
		pack_text = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True, enforce_sorted=False)

		pack_outputs, (hidden, cell) = self.rnn(pack_text, hidden)
		outputs, lens = nn.utils.rnn.pad_packed_sequence(pack_outputs)
		outputs = outputs.permute(1, 0, 2).contiguous()
		predictions = self.output(outputs)
		predictions = predictions.view(-1, predictions.shape[2])
		predictions = F.log_softmax(predictions, dim=1)
		return predictions, (hidden, cell)
