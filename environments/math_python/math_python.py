import verifiers as vf
from math_verify import verify
from verifiers.parsers.parser import Parser
from verifiers.parsers.think_parser import ThinkParser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, RewardFunc
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset
from verifiers.utils.tools import python


class MathRubric(Rubric):
    def __init__(
        self,
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: Parser | None = None,
    ):
        parser = parser or ThinkParser(extract_fn=extract_boxed_answer)
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_reward_func(self.correct_answer_reward_func)

    def correct_answer_reward_func(self, parser: Parser, completion: Messages, answer: str, **kwargs) -> float:
        """Reward function that checks if the final answer matches the expected answer."""
        try:
            response = parser.parse_answer(completion) or ""
            if response == "":
                return 0.0
            if verify(answer, response):
                return 1.0
            else:
                return 0.0
        except BaseException as e:
            return 0.0


def load_environment(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    max_turns: int = 5,
    **kwargs,
):
    dataset = load_example_dataset(dataset_name, dataset_split, n=num_train_examples)
    system_prompt = "Use python for all calculations (variables do not persist). Give your answer inside \\boxed{}."

    parser = vf.Parser(extract_fn=extract_boxed_answer)

    tool_rubric = vf.ToolRubric(tools=[python])
    math_rubric = MathRubric(parser=parser)
    rubric = vf.RubricGroup(parser=parser, rubrics=[tool_rubric, math_rubric])

    vf_env = vf.ToolEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        tools=[python],
        max_turns=max_turns,
        **kwargs,
    )

    return vf_env
