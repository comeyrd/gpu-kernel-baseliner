import numpy as np
from scipy import stats

def generate_book_z_scores(n=100):
    # Mapping gamma from 0.01 (1%) to 0.99 (99%) or 1.00
    gammas = np.linspace(0.01, 0.99, n) 
    
    z_scores = []
    for gamma in gammas:
        target_probability = (1 + gamma) / 2
        eta = stats.norm.ppf(target_probability)
        z_scores.append(eta)

    # C++ Formatting
    cpp_output = "const static double MEDIAN_Z_SCORES[100] = {\n    "
    for i, z in enumerate(z_scores):
        cpp_output += f"{z:.6f}"
        if i < n - 1:
            cpp_output += ", "
        if (i + 1) % 10 == 0 and i < n - 1:
            cpp_output += "\n    "
    
    cpp_output += "\n};"
    return cpp_output

print(generate_book_z_scores(100))