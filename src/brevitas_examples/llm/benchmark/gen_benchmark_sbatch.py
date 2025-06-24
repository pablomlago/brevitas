
from argparse import ArgumentParser
import math
import random

script_template = \
"""\
#!/bin/bash
#SBATCH --account=pmonteag
#SBATCH --job-name={job_name}       # job name for reference
#SBATCH --array=0-{num_tasks}       # TASK_ID for each task (max_range:--array=0-{num_tasks})
#SBATCH --nodes=1                   # job on a single node
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --cpus-per-task=32          # job name for referance
#SBATCH --gpus-per-task=4           # gpus per task
#SBATCH --nice={priority}           # Very low priority job
#SBATCH --time={allowed_time}       # time limit of the job in hh:mi:ss
#SBATCH --output=test_%A_%a.log     # Standard output and error log
{exclude_str}

echo $HOSTNAME
free -h
rocm-smi
source /home/pmonteag/.local/miniforge3/etc/profile.d/conda.sh
conda activate brv_clone_llm_rocm
cd /home/pmonteag/clones/brevitas

case $HOSTNAME in
    "radha0" )
        GPU_LIST=0,1,2,3,4,5,6,7
        GPUS_PER_PROCESS={num_gps_per_process}
        ;;
    *)
        GPU_LIST=0,1,2,3
        GPUS_PER_PROCESS={num_gps_per_process}
        ;;
esac

{unrolled_task_loop}
"""

exclude_template = \
"""\
#SBATCH --exclude={exclude_list}            # exclude select nodes from being utilised\
"""

loop_outer = \
"""\
case $SLURM_ARRAY_TASK_ID in
{tasks}
    *)  
        echo "Unknown TASK_ID: $SLURM_ARRAY_TASK_ID"
        ;;  
esac
"""

loop_inner = \
"""\
    "{task}" )
        {command_string} --gpus ${{GPU_LIST}} --num-gpus-per-process ${{GPUS_PER_PROCESS}} --start-index {start} --end-index {end}
        ;;
"""

# Example run:
# python gen_benchmark_sbatch.py --command-string "python static_gptq_rot_benchmark.py --config l3.2-1b-fused-had-static-float-int4-gptq.yml --results-folder static_wa_int4" --script-name l3_int4_static.sh --job-name l3_int4_static --num-jobs 3072 --num-gpus-per-process 1 --shuffle-seed 0
def gen_task_commands(command_string, num_tasks, jobs_per_task, shuffle_seed):
    start_and_end = [(i*jobs_per_task, (i+1)*jobs_per_task) for i in range(num_tasks)]
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        random.shuffle(start_and_end)
    case_list = []
    for i, s_a_e in enumerate(start_and_end):
        start, end = s_a_e
        case_str = loop_inner.format(task=i, command_string=command_string, start=start, end=end)
        case_list.append(case_str)
    return loop_outer.format(tasks="\n".join(case_list))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--command-string',
        type=str,
        required=True,
        help=
        'Benchmark script command prefix'
    )
    parser.add_argument(
        '--script-name',
        type=str,
        required=True,
        help=
        'Filename of the generated script'
    )
    parser.add_argument(
        '--job-name',
        type=str,
        required=True,
        help=
        'A name to give the job list on Slurm'
    )
    parser.add_argument(
        '--num-jobs',
        type=int,
        required=True,
        help=
        'Total number of jobs in the underlying benchmark YAML file'
    )
    parser.add_argument(
        '--num-gpus-per-process',
        type=int,
        required=True,
        help=
        'Number to GPUs required for each process'
    )
    parser.add_argument(
        '--num-gpus-per-node',
        type=int,
        default=8,
        help=
        'Number of "virtual" GPUs each node is expected to have. A higher value will schedule job serially (default: %(default)s)'
    )
    parser.add_argument(
        '--priority',
        type=int,
        default=100000,
        help=
        'The "nice" value to set for each job. The higher the value, the lower the priority (default: %(default)s)'
    )
    parser.add_argument(
        '--allowed-time',
        type=str,
        default="36:00:00",
        help=
        'A string denoting the maximum time allowed for each job (default: %(default)s)'
    )
    parser.add_argument(
        '--shuffle-seed',
        type=int,
        default=None,
        help=
        'The random seed to use to shuffle the jobs, if not set, no shuffling will be applied (default: %(default)s)'
    )
    parser.add_argument(
        '--exclusion-list',
        type=str,
        default=None,
        help=
        'A comma separated list of nodes that you want to exclude from being used, e.g., "radha0,radha5" (default: %(default)s)'
    )
    args = parser.parse_args()
    jobs_per_task = math.floor(args.num_gpus_per_node / args.num_gpus_per_process)
    num_tasks = math.ceil(args.num_jobs / jobs_per_task)
    unrolled_task_loop = gen_task_commands(args.command_string, num_tasks, jobs_per_task, args.shuffle_seed)
    exclude_string = "" if args.exclusion_list is None else exclude_template.format(exclude_list=args.exclusion_list)
    script_string = script_template.format(
        job_name=args.job_name,
        num_tasks=num_tasks-1,
        priority=args.priority,
        allowed_time=args.allowed_time,
        num_gps_per_process=args.num_gpus_per_process,
        unrolled_task_loop=unrolled_task_loop,
        exclude_str=exclude_string,
    )
    with open(args.script_name, "w") as f:
        f.write(script_string)