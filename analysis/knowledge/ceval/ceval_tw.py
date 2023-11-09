"""
C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models
https://arxiv.org/pdf/2305.08322.pdf

C-Eval is a comprehensive Chinese evaluation suite for foundation models.
It consists of 13948 multi-choice questions spanning 52 diverse disciplines
and four difficulty levels.

Homepage: https://cevalbenchmark.com/
"""
from lm_eval.base import MultipleChoiceTask

_CITATION = """
@article{huang2023ceval,
    title={C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models}, 
    author={Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and Fu, Yao and Sun, Maosong and He, Junxian},
    journal={arXiv preprint arXiv:2305.08322},
    year={2023}
}
"""


SUBJECTS = {
    "computer_network":"計算機網路",
    "operating_system":"操作系統",
    "computer_architecture":"計算機組成",
    "college_programming":"大學編程",
    "college_physics":"大學物理",
    "college_chemistry":"大學化學",
    "advanced_mathematics":"高等數學",
    "probability_and_statistics":"概率統計",
    "discrete_mathematics":"離散數學",
    "electrical_engineer":"註冊電器工程師",
    "metrology_engineer":"註冊劑量師",
    "high_school_mathematics":"高中數學",
    "high_school_physics":"高中物理",
    "high_school_chemistry":"高中化學",
    "high_school_biology":"高中生物",
    "middle_school_mathematics":"初中數學",
    "middle_school_biology":"初中生物",
    "middle_school_physics":"初中物理",
    "middle_school_chemistry":"初中化學",
    "veterinary_medicine":"獸醫學",
    "college_economics":"大學經濟學",
    "business_administration":"工商管理",
    "marxism":"馬克思主義基本原理",
    "mao_zedong_thought":"毛澤東思想和中國特色社會主義理論體系概論",
    "education_science":"教育學",
    "teacher_qualification":"教師資格",
    "high_school_politics":"高中政治",
    "high_school_geography":"高中地理",
    "middle_school_politics":"初中政治",
    "middle_school_geography":"初中地理",
    "modern_chinese_history":"近代史綱要",
    "ideological_and_moral_cultivation":"思想道德修養與法律基礎",
    "logic":"邏輯學",
    "law":"法學",
    "chinese_language_and_literature":"中國語言文學",
    "art_studies":"藝術學",
    "professional_tour_guide":"導遊資格",
    "legal_professional":"法律執業資格",
    "high_school_chinese":"高中語文",
    "high_school_history":"高中歷史",
    "middle_school_history":"初中歷史",
    "civil_servant":"公務員",
    "sports_science":"體育學",
    "plant_protection":"植物保護",
    "basic_medicine":"基礎醫學",
    "clinical_medicine":"臨床醫學",
    "urban_and_rural_planner":"註冊城鄉規劃師",
    "accountant":"註冊會計師",
    "fire_engineer":"註冊消防工程師",
    "environmental_impact_assessment_engineer":"環境影響評價工程師",
    "tax_accountant":"稅務師",
    "physician":"醫師資格"
}


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {Ceval-computer_network: Task, Ceval-clinical_medicine: Task}
    """
    return {f"Ceval-valid-{sub}": create_task(sub) for sub in SUBJECTS.keys()}


def create_task(subject):
    class Ceval(CevalSubject):
        def __init__(self):
            super().__init__(subject)

    return Ceval


class CevalSubject(MultipleChoiceTask):
    VERSION = 1
    DATASET_PATH = "lca0503/c-eval-zhtw"
    DATASET_NAME = None

    def __init__(self, subject):
        self.DATASET_NAME = subject
        super().__init__()

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc,self.dataset["val"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc,self.dataset["test"])

    def _format_subject(self, subject):
        words = subject.split("_")
        return " ".join(words)

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.DATASET_NAME
        description= f"以下是中國關於{SUBJECTS[subject]}的單項選擇題，請選出其中的正確答案。"
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            <prompt>
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            答案：
            """

            question = doc["question"].strip()
            choices = "".join(
                [f'{key}. {doc[key]}\n' for key in keys]
            )
            prompt = f"{question}\n{choices}答案："
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": ord(doc["answer"])-ord("A"),
        }

    def fewshot_examples(self, k, rnd):
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        # use the unchanged order of the dev set without sampling,
        return self._fewshot_docs[:k]

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
