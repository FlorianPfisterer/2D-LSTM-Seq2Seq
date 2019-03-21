from data.bleu_utils import calculate_bleu_score
import argparse

# define options
parser = argparse.ArgumentParser(description='eval_bleu.py')
parser.add_argument('--cand_file', type=str,
                    help='The path to the file containing the candidate translations (one per line).')
parser.add_argument('--ref_file', type=str,
                    help='The path to the file containing the reference translations (one per line).')
options = parser.parse_args()


def main():
    ref_file = options.ref_file
    cand_file = options.cand_file

    bleu_score = calculate_bleu_score(cand_file, ref_file)
    print("BLEU score: {}".format(bleu_score))


if __name__ == "__main__":
    main()
