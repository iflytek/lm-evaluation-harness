# TODO: Remove all TODO comments once the implementation is complete.
"""
Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

This is an evaluation harness for the HumanEval problem solving 
dataset described in the paper "Evaluating Large Language Models 
Trained on Code". It used to measure functional correctness for 
synthesizing programs from docstrings. It consists of 164 original 
programming problems, assessing language comprehension, algorithms, 
and simple mathematics, with some comparable to simple software 
interview questions.

https://github.com/openai/human-eval
"""
from lm_eval.base import Task, rf
from lm_eval.metrics import mean


_CITATION = """
@article{chen2021codex,
  title={Evaluating Large Language Models Trained on Code},
  author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
  year={2021},
  eprint={2107.03374},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
"""


class humaneval(Task):
    VERSION = 0
    DATASET_PATH = "openai_humaneval"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            # return self.dataset["test"]

            return map(self._process_doc, self.dataset['test'])

    def _process_doc(self, doc):
        
        return {
            "prompt" : doc["prompt"],
            "task_id" : doc["task_id"],
        } 

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return doc["prompt"]

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        target = ""
        return " " + target

    
    def fewshot_context(self, doc, num_fewshot, **kwargs):
        assert num_fewshot == 0, "humaneval is intended only for the zero-shot setting."
        kwargs["description"] = "Complete the following python code:"
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # TODO: Construct your language model requests with the request factory, `rf`,
        # and return them as an iterable.
        return rf.greedy_until(ctx, {"until": []})

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.
        # from human_eval.data import HUMAN_EVAL, write_jsonl
        # from human_eval.evaluation import evaluate_functional_correctness
        # import tempfile
        # import os.path as osp
        
        # predictions = [{
        #         'task_id': doc['task_id'],
        #         'completion' : results
        #     }]
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     out_dir = osp.join(tmp_dir, 'human_eval.json')
        #     write_jsonl(out_dir, predictions)

        #     k = [1, 10, 100]
        #     score = evaluate_functional_correctness(out_dir,
        #                       k,
        #                       n_workers=4,
        #                       timeout=3.0,
        #                       problem_file=HUMAN_EVAL)
        scoure = 0
        print('scoure', scoure, 'result', results)
        return {
            'scoure' : scoure
        }

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and an aggregation function as value which
        # determines how to combine results from each document in the dataset.
        # Check `lm_eval.metrics` to find built-in aggregation functions.
        return {
            'scoure' : mean}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {
            'scoure' : True}
