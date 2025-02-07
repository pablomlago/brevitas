from argparse import ArgumentParser
from argparse import Namespace
import datetime
from functools import reduce
import itertools
import multiprocessing
from multiprocessing import Queue
import os
import re
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import pandas as pd
import randomname as rn
import yaml

from brevitas_examples.llm.llm_args import create_llm_args_parser
from brevitas_examples.llm.llm_args import validate as validate_llm_args


def _parse_llm_log_results(job_log: str) -> Dict[str, Any]:
    # Find the line containing Float PPL number
    float_ppl_line = re.search(r"Float perplexity \((.*?)\): (\d+\.\d+)", job_log)
    float_ppl = float(float_ppl_line.group(2)) if float_ppl_line is not None else None
    # Find the line containing Quant PPL number
    quant_ppl_line = re.search(r"Quantized perplexity \((.*?)\): (\d+\.\d+)", job_log)
    quant_ppl = float(quant_ppl_line.group(2)) if quant_ppl_line is not None else None
    # Search for dictionary in log
    few_shot_eval_line = re.findall(r"({.*?})", job_log)
    # Retrieve last dictionary, in case other dictionaries were printed to the log
    few_shot_eval = eval(few_shot_eval_line[-1]) if len(few_shot_eval_line) > 0 else {}
    # Return the results from the log as a dictionary
    job_log_results = {
        "float_ppl": float_ppl,
        "quant_ppl": quant_ppl,
        **few_shot_eval,}
    return job_log_results


# Enable processing arguments for an arbitrary entrypoint
ENTRYPOINT_MAP = {
    "llm": {
        "args_parser": create_llm_args_parser(),
        "args_validate": validate_llm_args,
        "eval_metrics": ["float_ppl", "quant_ppl"],
        "log_parser": _parse_llm_log_results}}


def _make_float(value: Any) -> Any:
    try:
        float_value = float(value)
        return float_value
    except ValueError:
        return value


def run_args_bucket_process(
        entrypoint: str,
        id: int,
        num_processes: int,
        cuda_visible_devices: str,
        results_folder: str,
        max_num_retries: int,
        args_queue: Queue):
    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # Now import the entrypoint, thus making sure that CUDA_VISIBLE_DEVICES
    # was set before importing torch
    if entrypoint == "llm":
        from brevitas_examples.llm.main import quantize_llm
        main_entrypoint = quantize_llm
    else:
        main_entrypoint = None
        raise ValueError(f"Entrypoint {entrypoint} is not available")

    # Provide ballpark estimates of remaining time
    mean_running_time = 0
    num_runs = 0
    # Keep references to original stdout and stderr
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    # Iterate over the combinations launching the LLM entrypoint
    while True:
        try:
            # Extract an element of the queue of combinations
            args_tuple = args_queue.get(timeout=10.)
            if args_tuple is not None:
                args, extra_args, args_dict = args_tuple
            else:
                break
        except Exception:
            break
        print(
            f"Process: {id}, remaining combinations: {args_queue.qsize()}, remaining time: {'unknown' if num_runs == 0 else str(datetime.timedelta(seconds=int((args_queue.qsize() / num_processes + 1)*mean_running_time)))}"
        )
        job_name = f"{rn.get_name()}"
        job_folder = f"{results_folder}/{job_name}"
        # Create folder to store the results of the experiment
        os.mkdir(job_folder)
        # Save yaml file for reproducibility
        with open(f"{job_folder}/config.yaml", 'w') as f:
            yaml.dump(args_dict, f)
        # Enable reruning the process there was a crash
        num_retries = 0
        while num_retries < max_num_retries:
            stdout_file = open(f"{job_folder}/stdout.out", 'w')
            stderr_file = open(f"{job_folder}/stderr.out", 'w')
            # Redirect output to files
            sys.stdout = stdout_file
            sys.stderr = stderr_file
            # Record the wall-clock elapsed time when running the LLM entrypoint
            start_time = time.time()
            try:
                results, _ = main_entrypoint(args, extra_args)
                results = {k: _make_float(v) for k, v in results.items()}
            except Exception:
                # Print exception to stderr, so it can be checked in log
                print(traceback.format_exc(), file=sys.stderr)
                results = None
            end_time = time.time()
            # Restore stdout and stderr
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            # Calculate elapsed time
            running_time = end_time - start_time
            stdout_file.close()
            stderr_file.close()
            num_retries += 1
            # Dump information with the state and results of the run
            with open(f"{job_folder}/run_results.yaml", 'w') as f:
                yaml.dump({
                    "elapsed_time": running_time,
                    "status": "crashed" if results is None else "succesful",
                    "retry_number": num_retries,
                    **(results if results is not None else {})},
                          f)
            if results is not None:
                # Update mean running time and move to next combination
                num_runs += 1
                mean_running_time = mean_running_time * (
                    num_runs - 1) / num_runs + running_time / num_runs
                break


def parse_config_args(args: List[str]) -> Namespace:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=
        'Specify YAML with argument combinations (e.g., benchmark/benchmark_config.yml). Default: %(default)s.'
    )
    parser.add_argument(
        '--entrypoint',
        type=str,
        default="llm",
        choices=["llm"],
        help='Entrypoint to run. Default: %(default)s.')
    parser.add_argument(
        '--results-folder',
        type=str,
        default="./",
        help='Folder to store the experiment results. Default: %(default)s.')
    parser.add_argument(
        '--gpus',
        type=str,
        default="0",
        help=
        'Specify the identifiers of the GPUs to use in a comma-separated list. Default: %(default)s.'
    )
    parser.add_argument(
        '--num-gpus-per-process',
        type=int,
        default=1,
        help='Number of GPUs to each for running each argument combination. Default: %(default)s.')
    parser.add_argument(
        '--max-num-retries',
        type=int,
        default=1,
        help=
        'Number of retries for each argument combination in case a crash happens. Default: %(default)s.'
    )
    return parser.parse_args(args)


def parse_results(entrypoint: str, results_folder: str) -> pd.DataFrame:
    row_data_list = []
    job_config = None
    for entry in os.scandir(results_folder):
        if entry.is_dir() and entry.name not in ["__pycache__"]:
            # Get the identifier of the job
            job_name = os.path.basename(entry.path)
            # Retrieve the configuration from the YAML file
            with open(f"{results_folder}/{job_name}/config.yaml", 'r') as f:
                job_config = yaml.safe_load(f)
            with open(f"{results_folder}/{job_name}/run_results.yaml", 'r') as f:
                job_results = yaml.safe_load(f)
            # If the job was not succesful, try parsing the log
            if job_results["status"] == "crashed":
                # Load the log file
                with open(f"{results_folder}/{job_name}/stdout.out", 'r') as f:
                    job_log = f.read()
                    # Parse results from log
                    job_log_results = ENTRYPOINT_MAP[entrypoint]["log_parser"](job_log)
                # Manually populate the results
                job_results = {
                    "elapsed_time": job_results["elapsed_time"],
                    "status": job_results["status"],
                    "retry_number": job_results["retry_number"],
                    **job_log_results,}
            # Add entry to DataFrame
            row_data = {"job_id": job_name, **job_config, **job_results}
            row_data_list.append(row_data)
    if job_config is not None:
        # Columns are obtained by computing the union of the sets of keys in row_data_list, since,
        # for instance, some jobs might have crashed before completing the LM eval
        common_keys = ["job_id"] + list(job_config.keys()) + [
            "elapsed_time", "status", "retry_number"] + ENTRYPOINT_MAP[entrypoint]["eval_metrics"]
        common_keys_set = set(common_keys)
        columns = common_keys + list(
            reduce(lambda x, y: x.union(y), [set(row_data.keys()) for row_data in row_data_list
                                            ]).difference(common_keys_set))
        # Instantiate DataFrame to store the results
        df = pd.DataFrame(columns=columns)
        for row_data in row_data_list:
            # Fill missing columns with None
            df.loc[len(df)] = [row_data[key] if key in row_data else None for key in columns]
    else:
        raise ValueError(f"No experiments results were found in {results_folder}")
    return df


if __name__ == "__main__":
    # A CUDA error message is issued when changing CUDA_VISIBLE_DEVICES
    # if processes are started in fork mode
    multiprocessing.set_start_method('spawn')
    # Parse benchmark arguments
    script_args = parse_config_args(sys.argv[1:])
    # Retrieve the argument parser for the entrypoint
    entrypoint_parser: ArgumentParser = ENTRYPOINT_MAP[script_args.entrypoint]["args_parser"]
    validate_args: Callable = ENTRYPOINT_MAP[script_args.entrypoint]["args_validate"]
    # Instantiate directory for storing the results
    if not os.path.exists(script_args.results_folder):
        os.makedirs(script_args.results_folder)
    if script_args.config is not None:
        with open(script_args.config, 'r') as f:
            args_dict = yaml.safe_load(f)
    else:
        args_dict = {
            action.dest: [action.default] if action.choices is None else action.choices
            for action in entrypoint_parser._actions}
        del args_dict["help"]  # Config file cannot be specified via YAML
        del args_dict["config"]  # Config file cannot be specified via YAML
        # Save YAML in the results folder
        with open(f"{script_args.results_folder}/benchmark_config.yaml", 'w') as f:
            yaml.dump(args_dict, f)
    # Generate combinations of arguments
    args_keys, args_values = zip(*args_dict.items())
    # Extract the keys that are known to the argument parser
    parser_keys = set(action.dest for action in entrypoint_parser._actions)
    # Retrieve argument combinations that are valid for the LLM entrypoint
    q = Queue()
    for v in itertools.product(*args_values):
        args_dict = dict(zip(args_keys, v))
        try:
            # Separate the arguments that are know to the parser and the extra
            # arguments that are used, for instance, in rotation optimization
            args = {}
            extra_args = []
            for key, value in args_dict.items():
                if key in parser_keys:
                    args[key] = value
                else:
                    extra_args += [f"--{key.replace('_', '-')}", str(value)]
            args = SimpleNamespace(**args)
            # Only keep valid configurations
            validate_args(args, extra_args)
            q.put((args, extra_args, args_dict))
        except AssertionError as e:
            # Invalid configuration
            pass
    # Map the comma-separated string of GPU ids to a list
    cuda_available_devices = list(map(int, script_args.gpus.split(",")))
    # Number of argument combinations
    num_processes = len(cuda_available_devices) // script_args.num_gpus_per_process
    # Instantiate processes to run the argument combinations
    processes = []
    for i in range(num_processes):
        cuda_visible_devices = ",".join(
            map(str, cuda_available_devices[i:i + script_args.num_gpus_per_process]))
        process = multiprocessing.Process(
            target=run_args_bucket_process,
            args=(
                script_args.entrypoint,
                i,
                num_processes,
                cuda_visible_devices,
                script_args.results_folder,
                script_args.max_num_retries,
                q,
            ),
        )
        process.start()
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.join()
    # Parse results
    df = parse_results(script_args.entrypoint, script_args.results_folder)
    df.to_csv(f"{script_args.results_folder}/results.csv", index=False)
