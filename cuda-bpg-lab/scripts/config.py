"""Experiment names and short verification inputs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {"vector_add": "01_measurement", "memory_copy": "01_measurement", "global_copy": "02_global_memory",
               "transpose": "03_memory_hierarchy", "cache_working_set": "03_memory_hierarchy", "local_array": "03_memory_hierarchy",
               "reduction": "04_execution_config", "register_pressure": "04_execution_config",
               "integer_division": "05_instructions_control", "loop_unroll": "05_instructions_control", "branch_flow": "05_instructions_control",
               "host_transfer": "06_async_execution", "stream_pipeline": "06_async_execution", "concurrent_kernels": "06_async_execution", "gemm": "07_gemm"}
SMOKE = {"gemm": ["--rows", "37", "--cols", "65", "--depth", "33"], "transpose": ["--rows", "37", "--cols", "65"],
         "cache_working_set": ["--n", "65", "--working-set", "128", "--steps", "3"],
         "local_array": ["--n", "33", "--rounds", "3"],
         "reduction": ["--n", "4099", "--block", "128", "--items", "3"],
         "register_pressure": ["--n", "33", "--block", "32", "--rounds", "3"],
         "integer_division": ["--n", "33", "--rounds", "3", "--divisor", "8", "--signed", "1"],
         "loop_unroll": ["--n", "33", "--rounds", "9", "--chains", "4"],
         "branch_flow": ["--n", "65", "--rounds", "3"],
         "host_transfer": ["--n", "4099"],
         "stream_pipeline": ["--n", "4099", "--chunk", "257", "--streams", "3", "--buffers", "2", "--rounds", "3"],
         "concurrent_kernels": ["--n", "4099", "--jobs", "3", "--rounds", "17"]}
