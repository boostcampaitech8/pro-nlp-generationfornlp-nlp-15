from transformers import TrainerCallback

class EvalPredictCallback(TrainerCallback):
    def __init__(self, runner):
        self.runner = runner

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = self.runner._trainer

        epoch = int(state.epoch) if state.epoch is not None else 0

        self.runner.run_eval_prediction(
            trainer,
            tag=f"epoch_{epoch}",
        )