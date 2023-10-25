# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import MultipleChoiceTask
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
import numpy as np 

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


# TODO: Replace `NewTask` with the name of your Task.
class MedQA(MultipleChoiceTask):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "med_qa"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
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