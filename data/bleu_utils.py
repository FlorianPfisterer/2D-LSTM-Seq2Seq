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
    candidate = open(candidate_file, 'r').readlines()
    reference = open(reference_file, 'r').readlines()

    num_candidates = len(candidate)
    reference = reference[:num_candidates]
    assert len(reference) == len(candidate), 'Make sure there are at least as many references as candidates.'

    score = 0.
    for i in range(len(reference)):
        ref = reference[i].strip()
        cand = candidate[i].strip()
        score_i = sentence_bleu([ref.split()], cand.split(), weights=(0.5, 0.5))
        score += score_i

    score /= num_candidates
    return score
