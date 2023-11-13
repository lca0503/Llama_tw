import pandas as pd
from transformers import AutoTokenizer

tkr = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

file_path = '教育部4808個常用字.xls'

df = pd.read_excel(file_path)

zh_word_list = list(df["常用字"])

zh_num_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

zh_punct_list = ["。","，","、","；","：","？","！","「","」","『","』","―","～","…","（","）","《","》","〈","〉","．"]

special_token_list = [tkr.eos_token]

all_token_list = zh_word_list + zh_num_list + zh_punct_list + special_token_list

merged_list = []
for input_ids in tkr(all_token_list, add_special_tokens=False).input_ids:
    merged_list += input_ids

print(sorted(list(set(merged_list))))
