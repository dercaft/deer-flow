import os
import sys
from dotenv import load_dotenv
import datetime
import argparse
from pathlib import Path
from gaia.utils.gaia import GAIABenchmark
from functools import partial
import logging
import asyncio

load_dotenv()

from langgraph.graph import Graph
# --- 1. 全局日志配置函数 (必须添加这个函数) ---
def setup_global_logger(
    log_file_path="app.log", console_level=logging.INFO, file_level=logging.INFO
):
    """配置全局的根 logger"""
    root_logger = logging.getLogger()  # 获取根 logger

    # 清除所有现有的 handlers，以避免重复（仅在确定这是唯一配置点时）
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 根 logger 级别应为所有 handler 中的最低级别，或者一个合理的默认值
    # 例如，如果 console_level 是 DEBUG，file_level 是 INFO，则 root 应为 DEBUG
    root_logger.setLevel(min(console_level, file_level, logging.DEBUG))

    # --- 配置 File Handler (输出到文件) ---
    log_file_path_obj = Path(log_file_path)
    try:
        log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在
        file_handler = logging.FileHandler(log_file_path_obj, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # 如果文件创建失败，至少保证控制台输出可用
        print(
            f"错误：无法配置日志文件处理器 '{log_file_path_obj}': {e}", file=sys.stderr
        )

    # --- 配置 Stream Handler (输出到控制台) ---
    console_handler = logging.StreamHandler(sys.stdout)  # 输出到标准输出
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 此处不再调用 root_logger.info，因为它会在 setup 完成前尝试记录
    # print(f"全局 Logger 配置完成。控制台级别: {logging.getLevelName(console_level)}, 文件级别: {logging.getLevelName(file_level)} @ {log_file_path}")


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
    user_input: dict,  # should be format in dict: {"question": task['Question'], "file_name": task['file_name']}
    debug: bool = False,
    max_plan_iterations: int = 1,
    max_step_num: int = 3,
    enable_background_investigation: bool = True,
):
    """Run the agent workflow asynchronously with the given user input.

    Args:
        user_input: Dict with keys "question" and "file_name"
            "question": str, the question to be answered
            "file_name": str, the absolute path of the file to be used for the task
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
        "messages": [{"role": "user", "content": user_input["question"]}],
        "question": user_input["question"],
        "file_name": user_input["file_name"],
        "auto_accepted_plan": True,
        "enable_background_investigation": enable_background_investigation,
    }
    config = {
        "configurable": {
            "thread_id": "default",
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "mcp_settings": {
                "servers": {
                    "markitdown": {
                        "transport": "sse",
                        "url": "http://localhost:3001/sse",
                        "enabled_tools": ["convert_to_markdown"], # Update 0519
                        "add_to_agents": ["researcher", "coder"],
                    },
                    # "playwright-mcp": {
                    #     "transport": "sse",
                    #     "url": "http://localhost:8931/sse",
                    #     "enabled_tools": [
                    #         "browser_close",
                    #         "browser_resize",
                    #         "browser_console_messages",
                    #         "browser_handle_dialog",
                    #         "browser_file_upload",
                    #         "browser_install",
                    #         "browser_press_key",
                    #         "browser_navigate",
                    #         "browser_navigate_back",
                    #         "browser_navigate_forward",
                    #         "browser_network_requests",
                    #         "browser_pdf_save",
                    #         "browser_take_screenshot",
                    #         "browser_snapshot",
                    #         "browser_click",
                    #         "browser_drag",
                    #         "browser_hover",
                    #         "browser_type",
                    #         "browser_select_option",
                    #         "browser_tab_list",
                    #         "browser_tab_new",
                    #         "browser_tab_select",
                    #         "browser_tab_close",
                    #         "browser_generate_playwright_test",
                    #         "browser_wait_for"
                    #     ],
                    #     "add_to_agents": ["researcher", "coder"]
                    # }
                }
            },
        },
        "recursion_limit": 100,
    }

    last_message_cnt = 0
    raw_answer = None
    chat_history = []
    token_info = {}
    output=await graph.ainvoke(initial_state, config, stream_mode="values")
    print(output)
    raw_answer, chat_history, token_info=output["raw_answer"], output["chat_history"], output["token_info"]
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
    parser.add_argument("--idx", type=int, default=-1, help="Index to run")
    parser.add_argument("--task_id", type=str, default="", help="task ID to run")
    parser.add_argument("--save_result", action="store_true", help="Save results")
    parser.add_argument(
        "--log_path", type=str, default=default_log_path, help="Log file path"
    )
    args = parser.parse_args()

    # --- 2. 在 main 函数开始时调用全局日志配置 ---
    # 根据 args.debug 设置控制台日志级别
    console_log_level = logging.DEBUG if args.debug else logging.INFO
    setup_global_logger(
        log_file_path=args.log_path,
        console_level=console_log_level,
        file_level=logging.DEBUG,  # 文件日志记录 DEBUG 及以上信息，便于排查
    )

    # 此处 logger 已被全局配置，可以开始记录日志
    logger.info("主程序开始，参数已解析。")
    logger.debug(f"命令行参数: {args}")  # 仅当 console_log_level 为 DEBUG 时显示

    # Ensure directories exist
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    # Initialize the benchmark
    benchmark = GAIABenchmark(data_dir=args.data_dir, save_to=args.save_to)
    benchmark.load(force_download=False)

    # exit()  # Removed to allow benchmark to run
    # Get partial function
    run_agent_workflow_async_without_graph_input = partial(
        run_agent_workflow_async,
        debug=args.debug,
        max_plan_iterations=1,
        max_step_num=3,
        enable_background_investigation=True,
    )
    # Run the benchmark with only index 1
    result = await benchmark.run(
        run_agent_workflow_async=run_agent_workflow_async_without_graph_input,
        on=args.on,
        level=args.level,
        idx=[args.idx] if args.idx > -1 else None,  # Only run index 1
        save_result=args.save_result,
        task_id=args.task_id,
    )

    print(f"Benchmark results: {result}")


if __name__ == "__main__":
    asyncio.run(main())
    # logger.info("#"*40)
    # logger.info("State History:")
    # to_replay = None
    # config = {
    #     "configurable": {
    #         "thread_id": "default",
    #         "max_plan_iterations": 1,
    #         "max_step_num": 3,
    #     },
    #     "recursion_limit": 100,
    # }
    # for state in graph.get_state_history(config):
    #     logger.info("-"*80)
    #     logger.info(f"State: {state}")
    #     logger.info(f"Next: {state.next}")
    #     logger.info(f"Messages: {len(state.values['messages'])}")
    #     logger.info("-"*80)
        
