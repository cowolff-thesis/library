import numpy as np

def flatten_list(data):
    flat_list = []
    for item in data:
        flat_list.extend(item.values())
    return flat_list

def reverse_flatten_list_with_agent_list(flat_list, agent_names):
    reversed_data = []
    for i in range(0, len(flat_list), len(agent_names)):
        # Creating a dictionary for each set of lists corresponding to the agent names
        entry = {agent_names[j]: flat_list[i + j] for j in range(len(agent_names))}
        reversed_data.append(entry)
    return reversed_data

def normalize_batch_observations(batch_observations, min_values, max_values):
    """ Normalize a batch of observations to the range [0, 1] """
    # Ensure min_values and max_values are arrays and have the same shape as observations
    min_values = np.array(min_values, dtype=np.float32)
    max_values = np.array(max_values, dtype=np.float32)

    # Calculate the range, avoiding division by zero
    range_values = np.where((max_values > min_values), max_values - min_values, 1)
    
    # Normalize each observation in the batch
    normalized_batch = (batch_observations - min_values) / range_values
    
    return normalized_batch
