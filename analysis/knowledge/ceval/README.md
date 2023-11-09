# C-eval

An Chinese aggregated benchmark. Please refer to this [paper](https://arxiv.org/abs/2305.08322).


## Prepare dataset

The Traditional Chinese version can be found on [Huggingface](https://huggingface.co/datasets/lca0503/ceval-tw)
Here is the example which we translated the C-eval dataset into its Traditional Chinese version.
```
python3 translate_ceval.py
```


## Inference and Evaluate

We perform this analysis using the [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) package.
To correctly install the package, please follow the guide provided on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Here is the example to run C-eval benchmark.
For more information about running the [main.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py) script, please check the [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Task list: Ceval-valid-computer_network,Ceval-valid-operating_system,Ceval-valid-computer_architecture,Ceval-valid-college_programming,Ceval-valid-college_physics,Ceval-valid-college_chemistry,Ceval-valid-advanced_mathematics,Ceval-valid-probability_and_statistics,Ceval-valid-discrete_mathematics,Ceval-valid-electrical_engineer,Ceval-valid-metrology_engineer,Ceval-valid-high_school_mathematics,Ceval-valid-high_school_physics,Ceval-valid-high_school_chemistry,Ceval-valid-high_school_biology,Ceval-valid-middle_school_mathematics,Ceval-valid-middle_school_biology,Ceval-valid-middle_school_physics,Ceval-valid-middle_school_chemistry,Ceval-valid-veterinary_medicine,Ceval-valid-college_economics,Ceval-valid-business_administration,Ceval-valid-marxism,Ceval-valid-mao_zedong_thought,Ceval-valid-education_science,Ceval-valid-teacher_qualification,Ceval-valid-high_school_politics,Ceval-valid-high_school_geography,Ceval-valid-middle_school_politics,Ceval-valid-middle_school_geography,Ceval-valid-modern_chinese_history,Ceval-valid-ideological_and_moral_cultivation,Ceval-valid-logic,Ceval-valid-law,Ceval-valid-chinese_language_and_literature,Ceval-valid-art_studies,Ceval-valid-professional_tour_guide,Ceval-valid-legal_professional,Ceval-valid-high_school_chinese,Ceval-valid-high_school_history,Ceval-valid-middle_school_history,Ceval-valid-civil_servant,Ceval-valid-sports_science,Ceval-valid-plant_protection,Ceval-valid-basic_medicine,Ceval-valid-clinical_medicine,Ceval-valid-urban_and_rural_planner,Ceval-valid-accountant,Ceval-valid-fire_engineer,Ceval-valid-environmental_impact_assessment_engineer,Ceval-valid-tax_accountant,Ceval-valid-physician

```
python main.py --model=hf-causal-experimental --model_args="pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True,max_length=4096" \
--tasks=$TASK_LIST --num_fewshot=0 --batch_size=2 --output_path="results/7b-chat/ceval" --device cuda --no_cache
```

If you want to run Traditional Chinese version of the C-eval benchmark, replace the [lm-evaluation-harness/lm_eval/tasks/ceval.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/ceval.py) script with the [ceval_tw.py](ceval_tw.py) script.
