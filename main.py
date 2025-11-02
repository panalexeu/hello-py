import asyncio
import json
from collections.abc import Callable
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam
from docker.errors import DockerException
from dotenv import load_dotenv
from rich import print

from docker_shell import IsolatedShell
from submission_grader import grade_submission
from sys_prompt import SYS_PROMPT

MAX_TOKENS = 2048


class ShellExecResult(TypedDict):
    exit_code: str
    stdout: str
    stderr: str
    success: str


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def run_python_code(code: str, shell: IsolatedShell) -> ShellExecResult:
    return shell.exec_python(code, timeout=60 * 30)


def run_exec(command: str, shell: IsolatedShell) -> ShellExecResult:
    return shell.exec(command, timeout=60 * 30)


def submit_answer_tool(answer: str) -> bool:
    """
    Tool for submitting the final answer. As answer provide full file path for the submission.
    """
    return {"answer": answer, "submitted": True}


async def run_agent_loop(
        prompt: str,
        tools: list[ToolUnionParam],
        tool_handlers: dict[str, Callable[..., Any]],
        shell: IsolatedShell,
        max_steps: int = 15,
        model: str = "claude-haiku-4-5",
        verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )
        if response.stop_reason == "max_tokens":
            print(
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                "MAX_TOKENS, simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]

                    elif tool_name == "run_python":
                        if verbose:
                            print("\nInput:")
                            print("```")
                            print(tool_input['code'])
                            print("```")

                        result = handler(tool_input['code'], shell=shell)

                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")

                    elif tool_name == "exec":
                        if verbose:
                            print("\nInput:")
                            print("```")
                            print(tool_input['command'])
                            print("```")

                        result = handler(tool_input['command'], shell=shell)

                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")

                    else:
                        # Generic handler call
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use, conversation might be complete
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
        run_id: int,
        num_runs: int,
        prompt: str,
        tools: list[ToolUnionParam],
        tool_handlers: dict[str, Callable[..., Any]],
        f1_threshold: float,
        verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    # run each agent loop in a new isolated shell
    with IsolatedShell(image='ml-env') as shell:
        result = await run_agent_loop(
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            max_steps=10,
            verbose=verbose,
            shell=shell
        )

        res = 0
        success = False
        # if model provided some result
        if result:
            # try to retrieve the submission
            try:
                submission = shell.get_file(result)
                res = grade_submission(submission)

            # if submission path was not correctly provided, try to locate the file in container
            except (FileNotFoundError, DockerException) as _:
                if verbose:
                    print('Model did not correctly provide submission path')

                # try to find submission.csv using shell command
                find_result = shell.exec("find / -name 'submission.csv' -type f 2>/dev/null | head -n 1")
                path = find_result['stdout'].strip()

                # use found file as submission
                if path and find_result['success']:
                    if verbose:
                        print('However submission.csv was located in the container')

                    submission = shell.get_file(path)
                    res = grade_submission(submission)

            if res >= f1_threshold:
                success = True

    if success:
        print(f"✓ Run {run_id}: SUCCESS - Got f1 score: {res}")
    else:
        print(f"✗ Run {run_id}: FAILURE - Got {res}, expected f1 score >= {f1_threshold}")

    return run_id, success, res


async def main(concurrent: bool = True):
    tools: list[ToolUnionParam] = [
        {
            "name": "run_python",
            "description": "Run provided multiline python code. Use only this tool for running python code",
            "input_schema": {
                "type": "object",
                "properties": {"code": {"description": "Provided multiline python code"}},
                "required": ["code"]
            }
        },
        {
            "name": "exec",
            "description": "Execute provided command in the shell environment",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"description": "A shell command"}},
                "required": ["command"]
            }
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        }
    ]

    tool_handlers = {
        "run_python": run_python_code,
        "exec": run_exec,
        "submit_answer": submit_answer_tool
    }

    # Run the test 10 times and track success rate
    num_runs = 10
    prompt = SYS_PROMPT

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            f1_threshold=0.81,
            verbose=True,
        )
        for i in range(num_runs)
    ]

    # Run concurrently or sequentially based on the flag
    if concurrent:
        # Process results as they complete
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        # Run sequentially by awaiting each task in order
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Count successes
    successes = sum(bool(success) for _, success, _ in results)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    load_dotenv()  # to load envs from .env

    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=False))
