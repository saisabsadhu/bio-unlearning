from evals.base import Evaluator


class BioUnlearnEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("BioUnlearn", eval_cfg, **kwargs)
