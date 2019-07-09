import math
import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

from src.manager import PointerSoftmaxModelManager

src = Field(batch_first=True, include_lengths=True, lower=True)
trg = Field(batch_first=True, include_lengths=True, lower=True)

train_data, val, test = Multi30k.splits(exts=('.en', '.de'), fields=(src, trg))

# build vocab using train data only
src.build_vocab(train_data, min_freq=2)
trg.build_vocab(train_data, min_freq=2)

device = torch.device('cuda')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, val, test), batch_size=32, device=device)

pad_idx = trg.vocab.stoi['<pad>']

N_EPOCHS = 3

best_validation_loss = float('inf')

model = PointerSoftmaxModelManager(src_vocab=len(src.vocab), tgt_vocab=len(trg.vocab), pad_idx=pad_idx)

for epoch in range(N_EPOCHS):
    train_loss = model.train(train_iterator)
    valid_loss = model.evaluate(valid_iterator)

    if valid_loss < best_validation_loss:
        best_validation_loss = valid_loss
        print(f'| Epoch: {epoch + 1:03} | Train Loss: {train_loss:.3f} | Train PPL: '
              f'{math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
