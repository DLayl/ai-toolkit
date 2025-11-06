import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
from typing import Union, OrderedDict
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'


def _ensure_cuda_runtime_loaded():
    """Preload libcudart if it is packaged inside the Python environment.

    Some bundled CUDA wheels ship the runtime under site-packages/nvidia/..., which
    the dynamic loader does not scan by default. When flash-attn tries to import it
    later we can proactively dlopen the shared object so it becomes available.
    """

    if os.name != 'posix' or os.environ.get('AITOOLKIT_SKIP_CUDA_RUNTIME_FIX') == '1':
        return

    import ctypes
    import ctypes.util
    from pathlib import Path

    # If the runtime is already visible we can stop early.
    if ctypes.util.find_library('cudart'):  # returns absolute path if visible
        return

    search_roots = set()

    # Honour an explicit override first.
    override = os.environ.get('AITOOLKIT_CUDA_RUNTIME_PATH')
    if override:
        search_roots.add(Path(override))

    # Scan sys.path entries (site-packages, editable installs, etc.)
    for entry in list(sys.path):
        try:
            path_obj = Path(entry)
        except TypeError:
            continue
        # Add the entry and a couple of obvious parents that frequently host the libs.
        search_roots.add(path_obj)
        search_roots.add(path_obj.parent)

    # Also look beneath the virtualenv prefix if present.
    prefix = getattr(sys, 'prefix', None)
    if prefix:
        search_roots.add(Path(prefix))

    libc_name = 'libcudart.so.13'

    for root in search_roots:
        if not root.exists():
            continue
        try:
            matches = list(root.rglob(libc_name))
        except OSError:
            continue
        for candidate in matches:
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue

    # If we reach this point the runtime is still not visible; emit a gentle warning
    # so users know why flash-attn may fail to load later.
    if os.environ.get('AITOOLKIT_SUPPRESS_CUDA_RUNTIME_WARNING') != '1':
        sys.stderr.write(
            "[ai-toolkit] Warning: Could not locate libcudart.so.13 automatically. "
            "If flash-attn import fails, install the CUDA runtime or set "
            "AITOOLKIT_CUDA_RUNTIME_PATH to the directory containing libcudart.so.13.\n"
        )


_ensure_cuda_runtime_loaded()

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    args = parser.parse_args()
    
    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e


if __name__ == '__main__':
    main()
