import os
import subprocess
import sys
import argparse

def call_hipify_perl(input_file):
    # Call hipify-perl and capture its output
    result = subprocess.run(["hipify-perl", input_file], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"hipify-perl failed for {input_file}:\n{result.stderr}")
        return None
    return result.stdout


def baseliner_hipify(input_file):
    hipified = call_hipify_perl(input_file)
    lines = hipified.splitlines()
    modified_lines = []
    for i, line in enumerate(lines):
        line = line.replace('cuda', 'hip')
        line = line.replace('CUDA', 'HIP')
        line = line.replace('Cuda', 'Hip')
        modified_lines.append(line)
    hipified_baseliner = "\n".join(modified_lines) + "\n"
    try:
        with open(input_file, 'w') as f:
            f.write(hipified_baseliner)
        print(f"Success! Content saved back to {input_file}.")
    except IOError as e:
        print(f"Error: Could not write to file {input_file}: {e}")
        sys.exit(1)
        
def main():
    parser = argparse.ArgumentParser(
        description="Hipify a file in-place using hipify-perl and custom string replacements."
    )
    parser.add_argument(
        "filename",
        help="The path to the file to be processed (e.g., my_cuda_file.cpp)"
    )
    
    args = parser.parse_args()
    input_file = args.filename
    
    if not os.path.exists(input_file):
        print(f"Error: File not found at path: {input_file}")
        sys.exit(1)

    baseliner_hipify(input_file)

if __name__ == "__main__":
    main()