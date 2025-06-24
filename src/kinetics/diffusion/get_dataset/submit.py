import os
import subprocess
import argparse
from datetime import datetime


def submit_vasp_jobs(base_dir, group):
    """
    Recursively search and submit VASP calculation tasks in the specified directory

    Args:
        base_dir (str): Base directory path
        group (str): Calculation group name
    """
    # Record start time
    start_time = datetime.now()
    print(f"Task submission started: {start_time}")

    # Counters
    submitted = 0
    failed = 0

    # Traverse all subdirectories
    for root, dirs, files in os.walk(base_dir):
        if "run.sh" in files:
            try:
                # Check for necessary VASP input files
                required_files = ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]
                missing_files = [f for f in required_files if f not in files]

                if missing_files:
                    print(f"Warning: {root} is missing files: {', '.join(missing_files)}")
                    continue

                # Submit job
                cmd = ["qsub", "-g", group, "run.sh"]
                print(f"Submitting: {root}")
                result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)

                if result.returncode == 0:
                    job_id = result.stdout.strip()
                    print(f"Successfully submitted job {job_id} in {root}")
                    submitted += 1
                else:
                    print(f"Submission failed in {root}: {result.stderr}")
                    failed += 1

            except Exception as e:
                print(f"Error occurred in {root}: {str(e)}")
                failed += 1

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    print("\nTask submission summary:")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total duration: {duration}")
    print(f"Successfully submitted: {submitted}")
    print(f"Submission failed: {failed}")
    print(f"Total attempts: {submitted + failed}")


def main():
    parser = argparse.ArgumentParser(description='Batch submit VASP calculation tasks')
    parser.add_argument('base_dir', help='Base path containing VASP calculation directories')
    parser.add_argument('--group', required=True, help='Calculation group name ')

    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Error: Directory does not exist: {args.base_dir}")
        return

    submit_vasp_jobs(args.base_dir, args.group)


if __name__ == "__main__":
    main()
