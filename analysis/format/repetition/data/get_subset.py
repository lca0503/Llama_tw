import os
import random

import fasttext

N = 10000

random.seed(0)

with open("NeuLab-TedTalks.en-zh_tw.en", "r") as en_file:
    en_data = en_file.readlines()

with open("NeuLab-TedTalks.en-zh_tw.zh_tw", "r") as zh_file:
    zh_data = zh_file.readlines()

assert len(en_data) == len(zh_data)

lid_model = fasttext.load_model("lid.176.bin")

clean_en_data = []
clean_zh_data = []
for ed, zd in zip(en_data, zh_data):
    ed = ed.strip()
    zd = zd.strip()
    if len(ed) < 10 or len(zd) < 10:
        continue

    ed_lang = lid_model.predict(ed)
    zd_lang = lid_model.predict(zd)
    if ed_lang[0][0] != "__label__en" or zd_lang[0][0] != "__label__zh":
        continue

    clean_en_data.append(ed)
    clean_zh_data.append(zd)

assert len(clean_en_data) == len(clean_zh_data)
print(len(clean_en_data))

idx_list = list(range(len(clean_en_data)))

random.shuffle(idx_list)

select_idx = idx_list[:N]

os.makedirs("NeuLab-TedTalks", exist_ok=True)

with open("NeuLab-TedTalks/en-zh_tw.en", "w") as en_subsetfile:
    for idx in select_idx:
        en_subsetfile.write(clean_en_data[idx] + "\n")

with open("NeuLab-TedTalks/en-zh_tw.zh_tw", "w") as zh_subsetfile:
    for idx in select_idx:
        zh_subsetfile.write(clean_zh_data[idx] + "\n")
