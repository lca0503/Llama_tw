import json
import matplotlib.pyplot as plt


model_choice = [
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
    "Llama-2-70b-hf",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf"
]

plt.figure(figsize=(20, 10))

colors = ['b', 'g', 'r', 'c', 'm', 'y']

categories = {
    'STEM': [
        'abstract_algebra',
        'astronomy',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'electrical_engineering',
        'elementary_mathematics',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_mathematics',
        'high_school_physics',
        'high_school_statistics',
        'machine_learning'
    ],
    'Humanities': [
        'formal_logic',
        'high_school_european_history',
        'high_school_us_history',
        'high_school_world_history',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'moral_disputes',
        'moral_scenarios',
        'philosophy',
        'prehistory',
        'professional_law',
        'world_religions'
    ],
    'Social sciences': [
        'econometrics',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_microeconomics',
        'high_school_psychology',
        'human_sexuality',
        'professional_psychology',
        'public_relations',
        'security_studies',
        'sociology',
        'us_foreign_policy'
    ],
    'Other': [
        'anatomy',
        'business_ethics',
        'clinical_knowledge',
        'college_medicine',
        'global_facts',
        'human_aging',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'nutrition',
        'professional_accounting',
        'professional_medicine',
        'virology'
    ]
}

for k, v in categories.items():
    for mc, color in zip(model_choice, colors):
        acc_dict = {}
        with open(f"results/{mc}/mmlu") as cevalfile:
            ceval_results = json.load(cevalfile)
            for key, value in ceval_results["results"].items():
                task_name = key.split("hendrycksTest-")[-1]
                if task_name in v:
                    acc_dict[task_name] = value["acc"]
        values = [acc_dict[class_name] for class_name in list(acc_dict.keys())]
        classes = ["\n".join(class_name.split("_")) for class_name in list(acc_dict.keys())]
        plt.plot(classes, values, marker='o', color=color, label=mc)

    plt.legend(title="Models", loc="upper left")
    plt.xticks(rotation=45)
    plt.title(f"MMLU {k} results details")
    plt.ylabel("acc")
    plt.grid(True)
    plt.savefig(f"./mmlu_plot/mmlu_acc_{k}.png", bbox_inches='tight', pad_inches=0.2)
    plt.clf()
