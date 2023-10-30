# TODO: Remove all TODO comments once the implementation is complete.
"""
What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams
https://arxiv.org/pdf/2009.13081v1.pdf

Multiple choice question answering based on the United States Medical License Exams (USMLE). The dataset is collected from the professional medical board exams. It covers three languages: English, simplified Chinese, and traditional Chinese, and contains 12,723, 34,251, and 14,123 questions for the three languages, respectively.

Homepage: https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view
"""
from lm_eval.base import MultipleChoiceTask
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
import numpy as np 

# TODO: Add the BibTeX citation for the task.
_CITATION = """
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
"""

class MedQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "med_qa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        '''
        {
            "question": "经调查证实出现医院感染流行时，医院应报告当地卫生行政部门的时间是（　　）。", 
            "options": {
                "A": "2小时", 
                "B": "4小时内", 
                "C": "12小时内", 
                "D": "24小时内"
            }, 
            "answer": "24小时内", 
            "meta_info": "卫生法规", 
            "answer_idx": "D"
        }
        '''

        def format_example(doc, keys):
            """
            <prompt>
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            答案是：
            """

            question = doc["question"].strip()
            choices = "".join([f"{key}. {doc['options'][key]}\n" for key in doc["options"].keys()])
            prompt = f"题目：{question}\n{choices}答案是："
            return prompt

        keys = list(doc["options"].keys())
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": ord(doc["answer_idx"]) - ord("A"),
        }

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        kwargs = {}
        description= f"以下是关于医疗诊断的单项选择题，请直接给出正确答案的选项。" 
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, rnd=rnd, **kwargs)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]