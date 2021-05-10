import torch

from torch import nn

from matplotlib import pyplot as plt
from tqdm import tqdm


class Trainer(object):

	def __init__(self, model, criterion, optimizer) -> None:
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.train_losses = []
		self.eval_losses = []

	def fit(self, loader, n_epochs, eval_loader=None) -> nn.Module:
		for _ in tqdm(range(n_epochs)):
			train_loss = 0
			self.model.train()
			for batch in loader:
				loss = self.handle_batch(batch, train=True)
				train_loss += loss
			self.train_losses.append(train_loss)

			if eval_loader is not None:
				eval_loss = 0
				with torch.no_grad():
					self.model.eval()
					for batch in eval_loader:
						loss = self.handle_batch(batch, train=False)
						eval_loss += loss
				self.eval_losses.append(eval_loss)
			# print(f'{epoch + 1}: {train_loss}')
		return self.model

	def handle_model(self, inputs: torch.Tensor, outputs: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
		total_loss = 0
		hidden, cell = self.model.init_hidden()

		for index in range(inputs.shape[1]):
			curr_input = inputs[:, index]
			curr_input = curr_input.unsqueeze(1)
			input_length = torch.ones_like(length)
			input_length = input_length.squeeze()

			predictions, (hidden, cell) = self.model(curr_input, input_length, (hidden, cell))
			loss = self.criterion(predictions, outputs[:, index])
			total_loss += loss
		return total_loss

	def handle_batch(self, batch, train=True) -> float:
		inputs = batch['input']
		outputs = batch['output']
		length = batch['length']

		batch_loss = 0

		if train:
			self.optimizer.zero_grad()

			loss = self.handle_model(inputs, outputs, length)
			loss.backward()

			self.optimizer.step()
			batch_loss += loss.item()

		else:
			loss = self.handle_model(inputs, outputs, length)
			batch_loss += loss.item()
		return batch_loss

	def plot_loss_curve(self, eval=False) -> None:
		plt.plot(self.train_losses, label='Train')
		if eval:
			plt.plot(self.eval_losses, label='Eval')
		plt.xlabel('N epochs')
		plt.ylabel('Loss')
		plt.grid(True)
		plt.legend()
		plt.show()
