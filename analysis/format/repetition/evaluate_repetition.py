import pickle
from collections import Counter

import fire
from nltk import ngrams


def main(
    results_path: str=""
):
    with open(results_path, "rb") as f:
        data = pickle.load(f)

    for N in [4, 8, 12, 16, 20]:
        total_scores = 0
        total_num = 0

        for whole_sequence in data:
            ngs = [ng for ng in ngrams(whole_sequence, N)]
            counter = Counter(ngs)
            try:
                score = 1.0 - len(counter)/len(ngs)
                total_scores += score
                total_num += 1
            except:
                total_num += 1
        print("=" * 30)
        print("{}-gram: {:.3f}".format(N, total_scores / total_num))


if __name__ == "__main__":
    fire.Fire(main)
