import argparse
from mpi4py import MPI
import subprocess
import os
import numpy as np
import time

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser()
    # Important TALYS arguments redacted here for confidentiality
    parser.add_argument("command1", help="explanation1")
    parser.add_argument("command2", help="explanation2")
    args = parser.parse_args()

    # Initialize the MPI communicator
    comm = MPI.COMM_WORLD

    # Get the rank and size
    rank = comm.Get_rank()
    size = comm.Get_size()

    # The base command for running the TALYS program
    base_cmd = "talys"

    # Get the array job index from the environment variable, or default to 0
    array_job_index = 0

    # Load the attributes from a dat file
    with open('inputfile.dat', 'r') as f:
        data = f.readlines()
        attribute1_and2 = [list(map(int, item.split()[:2])) for item in data]
        idxs = [int(item.split()[2]) for item in data]

    # Create a checkpoint directory before the main loop
    if rank == 0:
        checkpoint_dir = os.path.join("output_folderlocation", f"checkpoint_{array_job_index}")
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        checkpoint_dir = os.path.join("output_folderlocation", f"checkpoint_{array_job_index}")

    # Distribute the attributes across the cores
    attributes_for_this_core = attribute1_and2[rank::size]
    idxs_for_this_core = idxs[rank::size]

    print(rank)

    # Run the command for each attributes
    outputs = []
    idxs_collected = []
    for (attribute1, attribute2), idx in zip(attributes_for_this_core, idxs_for_this_core):
        # Generate the input for TALYS
        # Important TALYS input parameters redacted here for confidentiality
        input = f"""##necessary inputs for TALYS
    """
        # Further TALYS-specific details omitted

        # Create a directory for the process and run the command there
        dir_name = f"process_{rank}"
        os.makedirs(dir_name, exist_ok=True)

        # Record the start time
        start_time = time.time()
        
        try:
            result = subprocess.run(base_cmd, input=input, shell=True, check=True, text=True, capture_output=True, cwd=dir_name, timeout=1800)
            output = f"Output for attribute1 {attribute1} and attribute2 {attribute2}: {result.stdout}"
            print(output)
            idxs_collected.append(idx)
            outputs.append(output)
        except subprocess.TimeoutExpired:
            print(f"Execution timed out for {attribute1} and attribute2 {attribute2}")
            output = f"Output for {attribute1} and attribute2 {attribute2}: timeout"
            outputs.append(output)
        except subprocess.CalledProcessError as e:
            print(f"Execution failed: {e}")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print(f"Output: {e.output}")
            print(f"Error output: {e.stderr}")
        # Record the end time
        end_time = time.time()

        # Calculate the execution time
        execution_time = end_time - start_time

        # Save the checkpoint after every loop
        output_filename = os.path.join(checkpoint_dir, f"checkpoint_{array_job_index}_{rank}.txt")
        with open(output_filename, 'a') as file:
            file.write(str(idx) + '\n')
            file.write(f"Execution time: {execution_time} seconds\n")
            file.write(output + '\n')

    # Gather the outputs to the process with rank 0
    all_outputs = comm.gather(outputs, root=0)
    all_idxs = comm.gather(idxs_collected, root=0)

    # Write the outputs to a file
    if rank == 0:
        # Flatten all_outputs and all_idxs
        all_outputs = [output for sublist in all_outputs for output in sublist]
        all_idxs = [idx for sublist in all_idxs for idx in sublist]

        # Pair each output with its corresponding index
        pairs = list(zip(all_idxs, all_outputs))

        # Sort the pairs based on the index
        pairs.sort()

        # Unzip the pairs back into sorted all_outputs and all_idxs
        all_idxs, all_outputs = zip(*pairs)

        output_filename = os.path.join("output_folderlocation", f"combined_output_{array_job_index}.txt")
        with open(output_filename, 'w') as file:
            for output, idx in zip(all_outputs, all_idxs):
                file.write(str(idx) + '\n')
                file.write(output + '\n')

if __name__ == "__main__":
    main()

