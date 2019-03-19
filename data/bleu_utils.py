from nltk.translate.bleu_score import sentence_bleu

"""
Contains utility functions for evaluating an NMT model based on the BLEU score.
"""


def calculate_bleu_score(candidate_file: str, reference_file: str) -> float:
    """
    Calculates the average BLEU score of the given files, interpreting each line as a sentence.
    Partially taken from https://stackoverflow.com/a/49886758/3918865.

    Args:
        candidate_file: the name of the file that contains the candidate sentences (hypotheses)
        reference_file: the name of the file that contains the reference sentences (targets)

    Returns:
        the average BLEU score
    """

    """"""
    reference = open(candidate_file, 'r').readlines()
    candidate = open(reference_file.candidate, 'r').readlines()
    assert len(reference) == len(candidate), 'The # of lines in the two files do not match'

    score = 0.
    for i in range(len(reference)):
        score += sentence_bleu([reference[i].strip().split()], candidate[i].strip().split())
    score /= len(reference)

    return score
