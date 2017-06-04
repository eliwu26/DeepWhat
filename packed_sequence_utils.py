import torch
import torch.nn as nn


# https://github.com/pytorch/pytorch/pull/1375/files
def get_last_step_indices(lengths):
    """A helper function for :func:`get_last_step_tensor`.

    The returned indices is used to select the last step of a :class:`PackedSequence` object.

    Arguments:
        lengths (list[int]): list of sequences lengths of each batch element.

    Returns:
        List of int containing the indices of the last step of each sequence.
    """
    n_lengths = len(lengths)
    rev_lengths = lengths[::-1]
    rev_lengths_sum = torch.LongTensor(rev_lengths).cumsum()
    return torch.LongTensor([(n_lengths - i - 1) * length + rev_lengths_sum[i]
                             for i, length in enumerate(rev_lengths)])

def get_last_step_tensor(packed_sequence, lengths):
    """Extract the last step of each sequence of a :class:`PackedSequence` object.

    It is useful for rnn's output.

    The returned Variable's data will be of size Bx*, where B is the batch size.

    Arguments:
        packed_sequence (PackedSequence): batch of sequences to extract last step
        lengths (list[int]): list of sequences lengths of each batch element.

    Returns:
        Variable containing the last step of each sequence in the batch.
    """
    indices = Variable(torch.LongTensor(get_last_step_indices(lengths)))
    if packed_sequence.data.data.is_cuda:
        indices = indices.cuda(packed_sequence.data.data.get_device())
    last_step = packed_sequence.data.index_select(0, indices)
    return last_step