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

categories_map = {
    'STEM': [
        'advanced_mathematics',
        'college_chemistry',
        'college_physics',
        'college_programming',
        'computer_architecture',
        'computer_network',
        'discrete_mathematics',
        'electrical_engineer',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_mathematics',
        'high_school_physics',
        'metrology_engineer',
        'middle_school_biology',
        'middle_school_chemistry',
        'middle_school_mathematics',
        'middle_school_physics',
        'operating_system',
        'probability_and_statistics',
        'veterinary_medicine'
    ],
    'Humanities': [
        'art_studies',
        'chinese_language_and_literature',
        'high_school_chinese',
        'high_school_history',
        'ideological_and_moral_cultivation',
        'law',
        'legal_professional',
        'logic',
        'middle_school_history',
        'modern_chinese_history',
        'professional_tour_guide'
    ],
    'Social Science': [
        'business_administration',
        'college_economics',
        'education_science',
        'high_school_geography',
        'high_school_politics',
        'mao_zedong_thought',
        'marxism',
        'middle_school_geography',
        'middle_school_politics',
        'teacher_qualification'
    ],
    'Other': [
        'accountant',
        'basic_medicine',
        'civil_servant',
        'clinical_medicine',
        'environmental_impact_assessment_engineer',
        'fire_engineer',
        'physician',
        'plant_protection',
        'sports_science',
        'tax_accountant',
        'urban_and_rural_planner'
    ]
}


for k, v in categories_map.items():
    for mc, color in zip(model_choice, colors):
        acc_dict = {}
        with open(f"results/{mc}/ceval_tw") as cevalfile:
            ceval_results = json.load(cevalfile)
            for key, value in ceval_results["results"].items():
                task_name = key.split("Ceval-valid-")[-1]
                if task_name in v:
                    acc_dict[task_name] = value["acc"]
        values = [acc_dict[class_name] for class_name in list(acc_dict.keys())]
        classes = ["\n".join(class_name.split("_")) for class_name in list(acc_dict.keys())]
        plt.plot(classes, values, marker='o', color=color, label=mc)
        
    plt.legend(title="Models", loc="upper left")
    plt.xticks(rotation=45)
    plt.title(f"CEVAL-TW {k} results details")
    plt.ylabel("acc")
    plt.grid(True)
    plt.savefig(f"./ceval_plot_tw/ceval_acc_{k}.png", bbox_inches='tight', pad_inches=0.2)
    plt.clf()
    
