import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import IWSLT, Multi30k

from src.manager import BaselineModelManager, PointerSoftmaxModelManager

src = Field(batch_first=True, include_lengths=True, lower=True)
trg = Field(batch_first=True, include_lengths=True, lower=True)

train_data, val, test = IWSLT.splits(exts=('.en', '.de'), fields=(src, trg),
                                     filter_pred=lambda x: max(len(vars(x)['src']), len(vars(x)['trg'])) <= 50)

# build vocab using train data only
src.build_vocab(train_data, min_freq=2, max_size=7700)
trg.build_vocab(train_data, min_freq=2, max_size=9500)

device = torch.device('cuda')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, val, test), batch_size=32, device=device)

pad_idx = trg.vocab.stoi['<pad>']

N_EPOCHS = 10

model = BaselineModelManager(src_vocab=src.vocab, tgt_vocab=trg.vocab, pad_idx=pad_idx)
# model = PointerSoftmaxModelManager(src_vocab=src.vocab, tgt_vocab=trg.vocab, pad_idx=pad_idx)

for epoch in range(N_EPOCHS):
    train_loss = model.train(train_iterator)
    valid_loss, valid_acc = model.evaluate(valid_iterator)
    model.save_checkpoint()

    print(f'| Epoch: {epoch + 1:03} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Val. Acc.: {valid_acc:.3f}')

model.load_checkpoint()
test_loss, test_accuracy = model.evaluate(test_iterator)
print(f'| Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f}')
