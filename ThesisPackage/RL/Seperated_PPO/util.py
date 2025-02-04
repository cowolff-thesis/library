import numpy as np
import torch

def flatten_list(data):
    flat_list = []
    for item in data:
        flat_list.extend(item.values())
    return flat_list

def flatten_for_agents(data, device):
    flat_dict = {key: [] for key in data[0].keys()}
    for item in data:
        for key in item:
            flat_dict[key].append(item[key])
    for key in flat_dict:
        flat_dict[key] = torch.tensor(np.array(flat_dict[key]), dtype=torch.float32).to(device)
    return flat_dict

def reverse_flatten_list_with_agent_list(flat_list, agent_names):
    reversed_data = []
    for i in range(0, len(flat_list), len(agent_names)):
        # Creating a dictionary for each set of lists corresponding to the agent names
        entry = {agent_names[j]: flat_list[i + j] for j in range(len(agent_names))}
        reversed_data.append(entry)
    return reversed_data