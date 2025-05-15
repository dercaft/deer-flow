import os
from dotenv import load_dotenv
import datetime
import argparse
from pathlib import Path
from gaia.utils.gaia import GAIABenchmark
from src.graph.builder import graph
from langgraph.graph import Graph
from functools import partial
import logging
import asyncio
load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))
default_data_dir = os.path.join(script_dir, "data/gaia")
default_save_to = os.path.join(
    script_dir,
    "results/gaia_results_{}.json".format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ),
)
default_log_path = os.path.join(
    script_dir,
    "logs/gaia_run_{}.log".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Logger 自身的最低级别，INFO 及以上都会被处理

async def run_agent_workflow_async(
    graph: Graph,
    user_input: str,
    debug: bool = False,
    max_plan_iterations: int = 1,
    max_step_num: int = 3,
    enable_background_investigation: bool = True,
):
    """Run the agent workflow asynchronously with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging
        max_plan_iterations: Maximum number of plan iterations
        max_step_num: Maximum number of steps in a plan
        enable_background_investigation: If True, performs web search before planning to enhance context

    Returns:
        Tuple[str, list, dict]: A tuple containing raw_answer, chat_history, and token_info
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Starting async workflow with user input: {user_input}")
    initial_state = {
        # Runtime Variables
        "messages": [{"role": "user", "content": user_input}],
        "question": user_input,
        "auto_accepted_plan": True,
        "enable_background_investigation": enable_background_investigation,
    }
    config = {
        "configurable": {
            "thread_id": "default",
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            # "mcp_settings": {
            #     "servers": {
            #         "mcp-github-trending": {
            #             "transport": "stdio",
            #             "command": "uvx",
            #             "args": ["mcp-github-trending"],
            #             "enabled_tools": ["get_github_trending_repositories"],
            #             "add_to_agents": ["researcher"],
            #         }
            #     }
            # },
        },
        "recursion_limit": 100,
    }

    last_message_cnt = 0
    raw_answer = None
    chat_history = []
    token_info = {}
    final_state = None

    async for s in graph.astream(
        input=initial_state, config=config, stream_mode="values"
    ):
        try:
            logger.info(f"Stream output: {type(s)}")
            if isinstance(s, dict):
                # Store the final state for extraction later
                final_state = s
                logger.info(f"Final state: {final_state}")  
                # Process message updates
                if "messages" in s:
                    if len(s["messages"]) <= last_message_cnt:
                        continue
                    last_message_cnt = len(s["messages"])
                    message = s["messages"][-1]
                    if isinstance(message, tuple):
                        print(message)
                    else:
                        message.pretty_print()
                if "raw_answer" in final_state:
                    raw_answer = final_state.get("raw_answer", None)
                if "chat_history" in final_state:
                    chat_history = final_state.get("chat_history", None)
                if "token_info" in final_state:
                    token_info = final_state.get("token_info", None)
            else:
                logger.info(f"Output: {s}")

        except Exception as e:
            logger.error(f"Error processing stream output: {e}")
            print(f"Error processing output: {str(e)}")

    logger.info("Async workflow completed successfully")
    logger.info(f"Raw answer: {raw_answer}")
    # check if return value exists
    if raw_answer is None:
        logger.error("Raw answer is None")
        return None, None, None

    # Return the answer node outputs
    return raw_answer, chat_history, token_info

async def main():
    parser = argparse.ArgumentParser(description="Run GAIA benchmark to test workflow")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help="Directory to store GAIA data",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=default_save_to,
        help="File to save results",
    )
    parser.add_argument(
        "--on",
        type=str,
        choices=["valid", "test"],
        default="valid",
        help="Dataset to run on",
    )
    parser.add_argument("--level", type=str, default="all", help="Level(s) to test")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--idx", type=int, default=1, help="Index to run")
    parser.add_argument("--save_result", action="store_true", help="Save results")
    parser.add_argument(
        "--log_path", type=str, default=default_log_path, help="Log path"
    )
    args = parser.parse_args()

    # Ensure directories exist
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    # Initialize the benchmark
    benchmark = GAIABenchmark(data_dir=args.data_dir, save_to=args.save_to)
    benchmark.load(force_download=False)

    print(graph.get_graph(xray=True).draw_mermaid())
    # exit()  # Removed to allow benchmark to run
    # Get partial function
    run_agent_workflow_sync_with_graph = partial(
        run_agent_workflow_async,
        graph=graph,
        debug=args.debug,
        max_plan_iterations=1,
        max_step_num=3,
        enable_background_investigation=True,
    )
    # Run the benchmark with only index 1
    result = await benchmark.run(
        run_agent_workflow_sync_with_graph=run_agent_workflow_sync_with_graph,
        on=args.on,
        level=args.level,
        idx=[args.idx],  # Only run index 1
        save_result=args.save_result,
    )

    print(f"Benchmark results: {result}")


if __name__ == "__main__":
    asyncio.run(main())
