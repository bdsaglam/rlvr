Status: DONE

Currently verifiers library logs prompt and completions such that it only displays first message and last message of agent's trajectory. See @tmp/verifiers/verifiers/trainers/grpo_trainer.py. But this makes it hard to debug and analyze the training process on W&B. I had implemented a function in my fork previously that logs the whole trajectory, see below. We need to do the same with the new version of the verifiers library. I'm not sure if the library provides a way to do this easily (callbacks, subclassing, etc.) But even if it doesn't, we can just copy GRPOTrainer and modify it to log the whole trajectory.

```sh
    def _log_interactions_to_wandb(
        self,
        inputs_to_log: list[dict],
        prompts_to_log: list[list[dict]],
        completions_to_log: list[dict],
        rewards_to_log: list[float],
        step: int,
    ) -> None:
        """Log interactions to wandb in both raw and formatted form.

        Args:
            inputs_to_log: List of input dictionaries
            prompts_to_log: List of prompt message dictionaries
            completions_to_log: List of completion message dictionaries
            rewards_to_log: List of reward values
            step: Current training step
        """
        import pandas as pd

        steps = [str(step)] * len(rewards_to_log)

        # Create a formatted table similar to rich table
        table_data = []
        for step, input, prompt, completion, reward in zip(
            steps,
            inputs_to_log,
            prompts_to_log,
            completions_to_log,
            rewards_to_log,
            strict=True,
        ):
            # Format input fields
            input_str = "\n\n".join(
                [
                    f"{k}: {str(v)[:1000]}"
                    for k, v in input.items()
                    if not isinstance(v, dict) and k not in ["prompt", "docs"]
                ]
            )

            table_data.append(
                {
                    "Step": step,
                    "Input": input_str,
                    "Prompt": _format_conversation(prompt),
                    "Completion": _format_conversation(completion),
                    "Reward": f"{reward:.2f}",
                }
            )

        table_df = pd.DataFrame(table_data)
        wandb.log({"interactions": wandb.Table(dataframe=table_df)})

```