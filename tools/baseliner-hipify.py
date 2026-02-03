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
        line = line.replace('.cu', '.hip')
        modified_lines.append(line)
    hipified_baseliner = "\n".join(modified_lines) + "\n"

    output_file = input_file
    if input_file.endswith('.cu'):
        output_file = input_file[:-3] + '.hip'
    output_file = output_file.replace('cuda', 'hip')
    output_file = output_file.replace('CUDA', 'HIP')
    output_file = output_file.replace('Cuda', 'Hip')
    output_file = output_file.replace('.cu', '.hip')   
    try:
        with open(output_file, 'w') as f:
            f.write(hipified_baseliner)
        
        if input_file != output_file and os.path.exists(input_file):
            os.remove(input_file)

        print(f"Success! Content saved to {output_file}.")
    except IOError as e:
        print(f"Error: Could not write to file {output_file}: {e}")
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