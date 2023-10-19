# TODO: Remove all TODO comments once the implementation is complete.
"""
Program Synthesis with Large Language Models
https://arxiv.org/pdf/2108.07732v1.pdf

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, 
a subset of the data has been hand-verified by us.

https://huggingface.co/datasets/mbpp
"""
from lm_eval.base import Task, rf
from lm_eval.metrics import mean

import contextlib
import io
import re
import signal

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
"""


class mbpp(Task):
    VERSION = 0
    DATASET_PATH = "mbpp"
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
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset['test'])

    def _process_doc(self, doc):
        
        return {
            "text" : doc["text"],
            "test_list" : '\n'.join(doc['test_list']),
        } 

    def doc_to_text(self, doc):
        return "Question:\n You are an expert Python programmer, and here is your task: {0} Your code should pass these tests:\n\n {1}".format(doc["text"], doc["test_list"])

    def doc_to_target(self, doc):
        target = ""
        return " " + target

    
    def fewshot_context(self, doc, num_fewshot, **kwargs):
        assert num_fewshot == 0, "mdpp is intended only for the zero-shot setting."
        kwargs["description"] = "Question:\n You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \n assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n Answer:\n [BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n[DONE] \n\n Question:\n You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \n assert is_not_prime(10) == True \n assert is_not_prime(35) == True \n Answer:\n [BEGIN]\n 'import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result' \n[DONE] \n\n Question:\n You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n Answer:\n [BEGIN]\n 'import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n[DONE] \n\n"
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, {"until": []})

    def _process_answer(self, text):
        text = text.strip()
        match = re.search(r"('\s*|)(\[DONE\]|DONE)", text)
        if match:
            text = text[:match.start()]
        match = re.search(r"(\[BEGIN\]|BEGIN)('\s*|)", text)
        if match:
            text = text[match.end():]
        text = text.strip()
        if text.startswith("'"):
            text = text[1:]
        if text.endswith("'"):
            text = text[:-1]
        return text

    def _process_test(self, test_case, pred):
        formatted = pred + '\n'
        formatted += test_case
        return formatted

    class WriteOnlyStringIO(io.StringIO):
        """StringIO that throws an exception when it's read from."""

        def read(self, *args, **kwargs):
            raise IOError

        def readline(self, *args, **kwargs):
            raise IOError

        def readlines(self, *args, **kwargs):
            raise IOError

        def readable(self, *args, **kwargs):
            """Returns True if the IO object can be read."""
            return False
    
    class redirect_stdin(contextlib._RedirectStream):  # type: ignore
        _stream = 'stdin'

    def swallow_io(self):
        stream = self.WriteOnlyStringIO()
        with contextlib.redirect_stdout(stream):
            with contextlib.redirect_stderr(stream):
                with self.redirect_stdin(stream):
                    yield

    def time_limit(self, seconds: float):

        def signal_handler(signum, frame):
            raise TimeOutException('Time out!')

        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, signal_handler)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        predictions = self._process_answer(results[0])
        programs = self._process_test(doc["test_list"], predictions)

        _pass = timeout = wrong_answer = failed = 0
        try:
            exec_globals = {}
            exec(programs, exec_globals)
            _pass += 1
        except TimeOutException:
            timeout += 1
        except AssertionError:
            wrong_answer += 1
        except BaseException:
            failed += 1

        return {
            '_pass' : _pass,
            'timeout' : timeout,
            'wrong_answer' : wrong_answer,
            'failed' : failed,
        }

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return {
            '_pass' : mean,
            'timeout' : mean,
            'wrong_answer' : mean,
            'failed' : mean,
        }

    def higher_is_better(self):
        return {
        '_pass' : True,
        'timeout' : True,
        'wrong_answer' : True,
        'failed' : True,
        }

class TimeOutException(Exception):
    pass