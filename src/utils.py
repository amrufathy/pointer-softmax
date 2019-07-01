import torch


def sequence_mask(lengths):
    """
    Creates a byte mask tensor where any element less than
        sequence length is 1 else 0

    https://github.com/jihunchoi/seq2seq-pytorch/blob/master/models/basic.py

    :param lengths: lengths of sequences
        shape: [batch x ]
    """
    max_length = lengths.max()
    seq_range = torch.arange(0, max_length).unsqueeze(0).type_as(lengths)
    lengths = lengths.unsqueeze(1)
    mask = torch.lt(seq_range, lengths)
    return mask
