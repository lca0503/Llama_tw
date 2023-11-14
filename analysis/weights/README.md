# Weight Analysis

In weight analysis, employing a histogram allows us to visualize the weights of each module and compare the distinctions between the Llama-2 and Llama-2-chat models. Additionally, we can utilize cosine similarity to identify the layers where the Llama-2 and Llama-2-chat models differ more significantly.

## Analyze

Firstly, it is imperative to partition the weights of each module to facilitate an efficient analysis of the two models. Here is an example:
```
python3 split_weights.py --model_name $MODEL_NAME --output_dir $OUTPUT_DIR
python3 split_weights.py --mdoel_name meta-llama/Llama-2-7b-hf --output_dir ./results
python3 split_weights.py --mdoel_name meta-llama/Llama-2-7b-chat-hf --output_dir ./results
```

Next, you can employ the [check_weights_hist.py](check_weights_hist.py) script to visualize the weights of each module, enabling a thorough comparison of the differences between the Llama-2 and Llama-2-chat models.
```
python3 check_weights_hist.py --model_name $MODEL_NAME --chat_model_name $CHAT_MODEL_NAME \
--input_dir $INPUT_DIR --output_dir $OUTPUT_DIR
python3 check_weights_hist.py --model_name meta-llama/Llama-2-7b-hf --chat_model_name meta-llama/Llama-2-7b-hf \
 --input_dir ./results --output_dir ./figures
```

Furthermore, you can utilize the [check_weights_cosine.py](check_weights_cosine.py) script to pinpoint the layers where the Llama-2 and Llama-2-chat models exhibit more substantial differences based on cosine similarity.
```
python3 check_weights_hist.py --model_name $MODEL_NAME --chat_model_name $CHAT_MODEL_NAME \
--input_dir $INPUT_DIR --output_dir $OUTPUT_DIR
python3 check_weights_cosine.py --model_name meta-llama/Llama-2-7b-hf --chat_model_name meta-llama/Llama-2-7b-hf \
 --input_dir ./results --output_dir ./figures
```
