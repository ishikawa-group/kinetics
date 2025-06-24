# ABO3 Materials Dataset Generation Documentation

## Table of Contents

1. [Scripts Overview](#1-scripts-overview)
2. [Execution Instructions](#2-execution-instructions)
3. [Function Documentation](#3-function-documentation)
4. [Directory Structure](#4-directory-structure)
5. [Example Usage Workflow](#5-example-usage-workflow)

## 1. Scripts Overview

A set of tools for generating ABO3 perovskite datasets from the Materials Project database, running VASP calculations, and collecting calculation results into a standardized JSON format.

- `fast_vasp.py`: Generates VASP input files for ABO3 materials from Materials Project
- `submit.py`: Batch submits VASP calculations to queue system
- `get_dataset_json.py`: Processes VASP outputs and creates a combined dataset

## 2. Execution Instructions

### 2.1 Generating VASP Inputs (fast_vasp.py)

```bash
python fast_vasp.py \
  --api-key YOUR_MP_API_KEY \
  --potcar-dir /path/to/POTCAR \
  --max-materials 10 \
  --crystal-system cubic \
  --vasp-path /path/to/vasp \
  --base-dir vasp_calculations \
  --preferred-potcar '{"Fe":"Fe_pv","O":"O"}' \
  --log-dir output_logs
```

Key Arguments:

- `--api-key`: Materials Project API key
- `--potcar-dir`: Directory containing POTCAR files
- `--max-materials`: Maximum number of materials to process
- `--crystal-system`: Filter by crystal system (cubic/tetragonal/etc)
- `--vasp-path`: Path to VASP executable
- `--preferred-potcar`: JSON string of preferred POTCAR types

### 2.2 Submitting Jobs (submit.py)

```bash
python submit.py /path/to/vasp_calculations --group YOUR_GROUP_NAME
```

Key Arguments:

- First argument: Base directory containing VASP calculations
- `--group`: Calculation group name for job scheduling

### 2.3 Creating Dataset (get_dataset_json.py)

```bash
python get_dataset_json.py /path/to/vasp_calculations [output_file.json]
```

Key Arguments:

- First argument: Directory containing completed VASP calculations
- Second argument (optional): Output JSON file path (default: dataset.json)

## 3. Function Documentation

### 3.1 VASP Input Generation (fast_vasp.py)

```python
def prepare_vasp_inputs(structure, material_id, formula, potcar_dir, vasp_path, base_dir):
    """
    Prepare all VASP input files
    Returns: Dictionary of file paths
    """

def process_abo3_materials(api_key, vasp_path, potcar_dir, max_materials):
    """
    Process ABO3 materials from Materials Project
    Returns: List of processed materials
    """
```

### 3.2 Job Submission (submit.py)

```python
def submit_vasp_jobs(base_dir, group):
    """
    Submit VASP calculations to queue system
    """
```

### 3.3 Dataset Creation (get_dataset_json.py)

```python
def parse_vasp_files(poscar_path, outcar_path, vasprun_path):
    """
    Parse VASP output files
    Returns: Dictionary with structure, energy, forces, stress
    """

def process_all_calculations(base_dir, output_file):
    """
    Process all calculations into single dataset
    Returns: Boolean indicating success
    """
```

## 4. Directory Structure

```
terminal/
├── vasp_calculations/
│   └── mp-xxx/
│       ├── INCAR
│       ├── KPOINTS
│       ├── POSCAR
│       ├── POTCAR
│       └── run.sh
├── output_logs/
│   └── mp_test_YYYYMMDD_HHMMSS.log
├── processed_materials.json
├── dataset.json
├── fast_vasp.py
├── submit.py
└── get_dataset_json.py
```

## 5. Example Usage Workflow

```bash
# 1. Generate VASP inputs for ABO3 materials
python fast_vasp.py --api-key YOUR_API_KEY --max-materials 5

# 2. Submit calculations to queue
python submit.py vasp_calculations --group YOUR_GROUP

# 3. After calculations complete, generate dataset
python get_dataset_json.py vasp_calculations dataset.json
```

For a quicker start, you can preconfigure default parameters in the parse_arguments() function, such as POTCAR path and VASP path etc. After that, you only need to run `python xx.py` to launch quickly.
