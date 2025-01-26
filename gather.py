import os
import re
import pandas as pd

# List of all directories to scan
directories = [
    './runs_test_greedy',
    './runs_test_greedy_2opt',
    './runs_test_sample',
    './runs_test_sample_2opt'
]

# Refined regex patterns to extract desired metrics
metric_pattern_gt = re.compile(r"val/gt_cost\s+\│\s+([\d\.]+)")
metric_pattern_solved = re.compile(r"val_solved_cost\s+\│\s+([\d\.]+)")
execution_time_pattern = re.compile(r"Execution time for .+?:\s+(\d+) seconds")

# List to store results
results = []

# Iterate over all directories and extract .txt files
for directory in directories:
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    gt_match = metric_pattern_gt.search(content)
                    solved_match = metric_pattern_solved.search(content)
                    time_match = execution_time_pattern.search(content)
                    if gt_match and solved_match and time_match:
                        val_gt_cost = float(gt_match.group(1))
                        val_solved_cost = float(solved_match.group(1))
                        execution_time = int(time_match.group(1))
                        results.append({
                            "File": f'{directory.split("./runs_test_")[-1]}-{file_path.split("/")[2].split("_log.txt")[0].split("test_")[-1]}',
                            "val/gt_cost": val_gt_cost,
                            "val_solved_cost": val_solved_cost,
                            "Execution Time (seconds)": execution_time
                        })
            except FileNotFoundError:
                print(f"File not found: {file_path}")

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv("validation_metrics_summary.csv", index=False)

# Print the results
print(results_df)