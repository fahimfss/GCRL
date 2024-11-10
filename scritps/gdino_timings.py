import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize dictionaries to store time data for each resolution
resolution_times = defaultdict(list)

# Load the data from the log file
with open('inference_results/logs.txt', 'r') as file:
    for line in file:
        log_data = json.loads(line.strip())
        resolution = log_data["resolution"]
        time = log_data["time"]
        if time > 0.2:
            print("Initial time")
            continue
        resolution_times[resolution].append(time)

# Calculate the mean and standard deviation of time for each resolution
mean_times = {}
std_errors = {}
for resolution, times in resolution_times.items():
    mean_times[resolution] = np.mean(times)
    std_errors[resolution] = np.std(times)

# Plot a bar chart with error bars for each resolution
plt.figure(figsize=(8, 6))
bars = plt.bar(
    mean_times.keys(), 
    mean_times.values(), 
    yerr=std_errors.values(), 
    capsize=5, 
    color=['skyblue', 'lightgreen', 'orange']  # lighter colors
)
plt.xlabel("Resolution")
plt.ylabel("Time (s)")
plt.title("Grounding Dino Inference Time by Resolution")

# Add average time values on top of each bar
for bar, avg_time in zip(bars, mean_times.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        bar.get_height() - bar.get_height()/12, 
        f'{avg_time:.5f}', 
        ha='center', 
        va='bottom',
        fontsize=10, 
        color='black'
    )

# Save the plot as an image file
plt.savefig("time_with_error_plot.png")
plt.close()