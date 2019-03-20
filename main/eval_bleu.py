from data.bleu_utils import calculate_bleu_score


def main():
    ref_file = 'data/iwslt14_small/tgt-val.txt'
    cand_file = 'preds-100.txt'

    bleu_score = calculate_bleu_score(cand_file, ref_file)
    print("BLEU score: {}".format(bleu_score))


if __name__ == "__main__":
    main()
