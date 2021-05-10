import argparse

from functools import partial

import torch

from torch import nn
from torch.utils.data import DataLoader

from utils import seed_everything
from utils import collate_fn
from utils import process, post_process
from dataset import ProcessText, LMDataset
from model import RNN
from trainer import Trainer
from infer import infer


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Hyper parameters for NN')

	parser.add_argument('--batch_size', action='store', type=int, required=True, help='Batch size to train')
	parser.add_argument('--special_chars', action='store', type=str, required=True, help='Chars to split sentence')
	parser.add_argument('--data_path', action='store', type=str, required=True, help='Path to train data')
	parser.add_argument('--embedding_size', action='store', type=int, required=True, help='Embedding size for NN')
	parser.add_argument('--hidden_size', action='store', type=int, required=True, help='Hidden size for NN')
	parser.add_argument('--n_layers', action='store', type=int, required=True, help='Number of layers for NN')
	parser.add_argument('--dropout', action='store', type=float, required=True, help='Drop out ratio for NN')
	parser.add_argument('--learning_rate', action='store', type=float, required=True, help='Learning rate to train NN')
	parser.add_argument('--n_epochs', action='store', type=int, required=True, help='Number epochs to train NN')
	parser.add_argument('--seed', action='store', type=int, required=False, help='Random seed for NN', default=241)

	args = parser.parse_args()

	seed_everything(args.seed)

	post_process = partial(post_process, special_chars=args.special_chars)

	text_process = ProcessText(sent_split_chars=args.special_chars,
							   pre_processing_func=process,
							   post_processing_func=post_process)

	dataset = LMDataset(args.data_path, text_process)

	config = {
		'vocab_size': len(dataset.word2id),
		'bs': args.batch_size,
		'emb_size': args.embedding_size,
		'hid_size': args.hidden_size,
		'n_layers': args.n_layers,
		'dropout': args.dropout,
		'pad_idx': dataset.word2id['<pad>']
	}

	collate_fn = partial(collate_fn, pad_value=dataset.word2id['<pad>'])
	train_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

	lm_model = RNN(config)
	criterion = nn.NLLLoss(ignore_index=config['pad_idx'])
	optimizer = torch.optim.Adam(lm_model.parameters(), lr=args.learning_rate)

	trainer = Trainer(lm_model, criterion, optimizer)
	lm_model = trainer.fit(train_loader, n_epochs=args.n_epochs)
	trainer.plot_loss_curve()

	result = infer(lm_model, dataset, input_str='три девицы', max_len=10)
	print(f'Infer result: {result}')
