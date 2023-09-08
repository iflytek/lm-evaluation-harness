import os
import sacrebleu
import datasets
from rouge_chinese import Rouge
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
import jieba

class UserDataGen(Task):
    VERSION = 0

    def __init__(self):
        assert self.DATA_LOCAL_PATH is not None
        _, ext = os.path.splitext(self.DATA_LOCAL_PATH)
        if ext == '.jsonl':
            ext = '.json'
        if (ext == '.json' or ext == '.csv' or ext == '.parquet'):
            self.dataset = datasets.load_dataset(ext[1:], data_files=self.DATA_LOCAL_PATH)
        self._training_docs = None
        self._fewshot_docs = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        raise NotImplementedError

    def validation_docs(self):
        raise NotImplementedError

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["train"])

    def _process_doc(self, doc):
        # print(doc)
        instruction = doc['instruction']
        input = doc['input']
        if input is not None and input !="":
            instruction = instruction+'\n'+input
        out_doc = {
            "query": "### Instruction:\n{instruction}\n\n### Response:".format_map({'instruction':instruction}),
            "answer": doc['output'],
        }
        return out_doc

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        description = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)
    
    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["answer"].strip()

    def construct_requests(self, doc, ctx):
        completion = rf.greedy_until(ctx, {"until": ['\n\n', '### Instruction']})
        return completion

    def process_results(self, doc, results):
        completion = results[0].strip()
        ref = doc['answer']
        # print(doc["query"])
        # print(completion)
        # print(ref)
        references = ' '.join(jieba.cut(ref))
        predictions = ' '.join(jieba.cut(completion))
        # BLEU
        bleu_score = self.bleu([[references]], [predictions])
    
        # ROUGE-N
        if predictions == '' or predictions == None:
            rouge1_score = rouge2_score = rougeL_score = 0
        else:
            rouge_results = self.rouge(references, predictions)
            # ROUGE-1
            rouge1_score = rouge_results[0]["rouge-1"]['r']
            # ROUGE-2
            rouge2_score = rouge_results[0]["rouge-2"]['r']
            # ROUGE-L
            rougeL_score = rouge_results[0]["rouge-l"]['r']

        return {
            "bleu-4": bleu_score,
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougeL": rougeL_score,
        }

    def aggregation(self):
        return {
            "bleu-4": mean,
            "rouge1": mean,
            "rouge2": mean,
            "rougeL": mean,
        }

    def higher_is_better(self):
        return {
            "bleu-4": True,
            "rouge1": True,
            "rouge2": True,
            "rougeL": True,
        }

    def bleu(self, refs, preds):
        score = sacrebleu.corpus_bleu(
            preds,
            refs,
            smooth_method="exp",
            smooth_value=0.0,
            force=False,
            lowercase=False,
            use_effective_order=False,
        ).score
        return score/100

    def rouge(self, refs, preds):
       rouge = Rouge()
       scores = rouge.get_scores(preds, refs)
       return scores