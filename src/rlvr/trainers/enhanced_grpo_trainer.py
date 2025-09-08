"""Enhanced GRPO Trainer with trajectory logging for W&B."""

import logging
from typing import Dict, Optional

import pandas as pd
from peft import PeftConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from verifiers import Environment
from verifiers.trainers.grpo_config import GRPOConfig
from verifiers.trainers.grpo_trainer import GRPOTrainer

import wandb

from ..utils.logging_helpers import (
    extract_trajectory_stats,
    format_conversation,
)


class EnhancedGRPOTrainer(GRPOTrainer):
    """
    Enhanced GRPO Trainer that logs conversation trajectories to W&B.

    This trainer extends the base GRPOTrainer to provide comprehensive logging
    of multi-turn conversations including all tool interactions, making it easier
    to debug training issues in multi-step environments.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        env: Environment,
        args: GRPOConfig,
        processing_class: PreTrainedTokenizerBase,
        callbacks: Optional[list] = None,
        optimizers: tuple = (None, None),
        peft_config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            env=env,
            args=args,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

        self.logger = logging.getLogger(__name__)

    def _log_trajectories_to_wandb(self, step: int) -> None:
        """
        Log conversation trajectories to W&B with comprehensive formatting.
        Uses the existing _textual_logs data but formats it nicely.

        Args:
            step: Current training step
        """
        if not self._textual_logs["prompt"]:
            return

        # Prepare data for the interactions table
        table_data = []
        # Log up to 5 samples
        num_samples = min(5, len(self._textual_logs["prompt"]))

        for i in range(num_samples):
            try:
                # Format completion
                completion = list(self._textual_logs["completion"])[i]
                formatted_completion = format_conversation(completion, max_length=10000)

                # Format rewards

                # Extract simple stats
                stats = extract_trajectory_stats(completion)

                # Create the table row
                row_data = {
                    "step": step,
                    "sample": i + 1,
                    "prompt": list(self._textual_logs["prompt"])[i],
                    "completion": formatted_completion,
                    "trajectory_length": stats.get("total_length", 0),
                }

                # Add individual reward components as separate columns
                for key, values in self._textual_logs["rewards"].items():
                    if i < len(values):
                        row_data[f"reward_{key}"] = list(values)[i]
                


                table_data.append(row_data)

            except Exception as e:
                self.logger.warning(f"Error formatting trajectory {i}: {e}")
                continue

        if table_data:
            df = pd.DataFrame(table_data)
            wandb.log({"trajectories": wandb.Table(dataframe=df)})
            self.logger.info(f"Logged {len(table_data)} trajectories to W&B at step {step}")

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log method to add enhanced trajectory logging.

        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for timing
        """
        # Add our enhanced logging if we're on the main process and using W&B
        if (
            self.accelerator.is_main_process
            and self.args.report_to
            and "wandb" in self.args.report_to
            and wandb.run is not None
        ):
            # Log trajectories using the existing _textual_logs data
            self._log_trajectories_to_wandb(self.state.global_step)

        # Call parent log method last as it clears the textual logs
        super().log(logs, start_time)
