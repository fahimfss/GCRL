import json

def read_file_to_dicts(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Example usage
file_path = '/home/fahim/Projects/jsac_rlc/RL-Chemist/inference_test_data/info_1.txt'  # Replace with your actual file path
data = read_file_to_dicts(file_path)
print(len(data))
print(data[0]['prompt'])