from torchtext.data.dataset import Dataset
from torchtext.data.batch import Batch
from typing import List

"""
Contains utility functions for working with any NMT dataset, based on torchtext.
"""


def create_homogenous_batches(dataset: Dataset, max_batch_size: int,
                              key_fn=lambda ex: -len(ex.src),
                              filter_fn=lambda ex: len(ex.src) > 0) -> List[Batch]:
    """
    Creates a list of batches such that for each batch b it holds that:
        - b contains at least one and at most max_batch_size examples
        - for any two examples e1, e2 in b: key_fn(e1) == key_fn(e2)
        - b does not contain any example e for which filter_fn(e) == False
    In addition, the batches b are sorted by (increasing) key_fn(e) for any e in b.

    Args:
        dataset: the dataset to take the batches from
        max_batch_size: how many examples one batch may contain at the most
        key_fn: function of type (Example) -> int, that is used to sort the batches. Each batch will only have examples
            that all have exactly the same key (e.g. source sentence length).
        filter_fn: function of type (Example) -> bool, that is used to filter the examples. No example e with
            filter_fn(e) == False will be contained in any batch.

    Returns: a list of batches with the above properties
    """
    sorted_examples = sorted(dataset.examples, key=key_fn)

    same_key_blocks = []
    previous_key = -1
    current_block = []

    for example in sorted_examples:
        if not filter_fn(example):
            continue

        key = key_fn(example)
        if previous_key == -1 or key != previous_key:
            previous_key = key
            # start a new block
            if len(current_block) > 0:
                same_key_blocks.append(current_block)
                current_block = []
        # append current example to corresponding block
        current_block.append(example)

    # append last block
    if len(current_block) > 0:
        same_key_blocks.append(current_block)

    # split up blocks in batches of size at most max_batch_size
    batches = []
    for block in same_key_blocks:
        i = 0
        while i < len(block):
            # take the next at most max_batch_size examples from this block
            data = block[i:i + max_batch_size]
            batches.append(Batch(data=data, dataset=dataset))
            i += len(data)

    return batches
