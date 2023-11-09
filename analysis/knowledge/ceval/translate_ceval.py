import os

import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm


def main():
    subset_name = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine', 'college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography', 'modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history', 'civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']

    translator = GoogleTranslator(source="auto", target="zh-TW")
    
    for sp in ["dev", "val"]:
        os.makedirs(f"./ceval-zhTW/{sp}", exist_ok=True)
        for sn in tqdm(subset_name, "Processing: "):
            df = pd.read_csv(f"./data/{sp}/{sn}_{sp}.csv")
            df["question"] = df["question"].apply(translator.translate)
            df["A"] = df["A"].apply(translator.translate)
            df["B"] = df["B"].apply(translator.translate)
            df["C"] = df["C"].apply(translator.translate)
            df["D"] = df["D"].apply(translator.translate)
            if sp == "dev":
                df["explanation"] = df["explanation"].apply(translator.translate)
            df.to_csv(f"./ceval-tw/{sp}/{sn}.csv", index=False)

            
if __name__ == '__main__':
    main()
