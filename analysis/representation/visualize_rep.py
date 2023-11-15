import os

import fire
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from umap import UMAP


def main(
    model_name: str="",
    input_dir: str="",
    output_dir: str="./figures",
):
    plt.figure(figsize=(10, 10))
    
    if "7b" in model_name:
        layer_num = 32 + 1
    elif "13b" in model_name:
        layer_num = 40 + 1
    elif "70b" in model_name:
        layer_num = 80 + 1
    else:
        raise NotImplementedError

    os.makedirs(f"{output_dir}/{model_name}/", exist_ok=True)

    for idx in range(layer_num):
        en_rep = np.load(f"./{input_dir}/{model_name}/en/layer-{idx}.npy")
        zh_rep = np.load(f"./{input_dir}/{model_name}/tw/layer-{idx}.npy")
        
        reps = np.concatenate((en_rep, zh_rep), axis=0)
    
        reducer = UMAP(random_state=42)
        reduce_reps = reducer.fit_transform(reps)

        plt.scatter(reduce_reps[:en_rep.shape[0], 0], reduce_reps[:en_rep.shape[0], 1], color='green', label='en', s=2)
        plt.scatter(reduce_reps[en_rep.shape[0]:, 0], reduce_reps[en_rep.shape[0]:, 1], color='orange', label='zh_tw', s=2)
        
        plt.legend(title="Languages", loc="upper left")
        plt.title(f"{model_name} layer-{idx} representation visualization (NeuLab-TedTalks)")
        plt.savefig(f"{output_dir}/{model_name}/layer-{idx}.png", bbox_inches='tight', pad_inches=0.2)
        plt.clf()
    print("Done!!!")


if __name__ == "__main__":
    fire.Fire(main)
