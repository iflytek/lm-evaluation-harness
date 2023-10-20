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
            "test_list" : '\n'.join([item.strip() for item in doc['test_list']]),
            "code" : doc["code"]
        } 

    def doc_to_text(self, doc):
        return "Question: here is your task: {0} Your code should pass these tests:\n\n {1}".format(doc["text"], doc["test_list"])

    def doc_to_target(self, doc):
        target = ""
        return "Answer:\n [BEGIN]\n '{0}' \n[DONE] \n\n".format(doc["code"])

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

    @contextlib.contextmanager
    def swallow_io(self):
        stream = self.WriteOnlyStringIO()
        with contextlib.redirect_stdout(stream):
            with contextlib.redirect_stderr(stream):
                with self.redirect_stdin(stream):
                    yield

    @contextlib.contextmanager
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
        print(programs)
        try:                
            exec_globals = {}
            with self.swallow_io():
                with self.time_limit(2):
                    exec(programs, exec_globals)
            _pass += 1
        except TimeOutException:
            timeout += 1
        except AssertionError:
            wrong_answer += 1
        except BaseException:
            failed += 1

        return {
            'pass' : _pass,
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
            'pass' : mean,
            'timeout' : mean,
            'wrong_answer' : mean,
            'failed' : mean,
        }

    def higher_is_better(self):
        return {
        'pass' : True,
        'timeout' : True,
        'wrong_answer' : True,
        'failed' : True,
        }

class TimeOutException(Exception):
    pass