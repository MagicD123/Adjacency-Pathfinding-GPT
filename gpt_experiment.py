import json
import heapq
from openai import OpenAI
from tqdm import tqdm
import os
import time
from collections import deque

client = OpenAI(api_key="")

TWO_STEP_PROCESS = True
USE_TEXT_DESCRIPTION = False  

def read_json(filename: str):
    """
    Read the specified JSON file and return its contents.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file '{filename}': {e}")
        return None

def bfs_shortest_path(adjacency_matrix, start, end):
    """
    Calculate the shortest path using Breadth-First Search (BFS).
    """
    size = len(adjacency_matrix)
    visited = [False] * size
    queue = deque([(start, [start])])
    visited[start] = True

    while queue:
        current_node, path = queue.popleft()
        if current_node == end:
            return [node + 1 for node in path], len(path) - 1  # Convert to 1-based index

        for neighbor, connected in enumerate(adjacency_matrix[current_node]):
            if connected and not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, path + [neighbor]))

    return [], -1  # If no path exists

def adjacency_matrix_to_text(adjacency_matrix):
    """
    Convert an adjacency matrix to a textual description.
    """
    description = ""
    for i, row in enumerate(adjacency_matrix):
        connections = [str(j + 1) for j, val in enumerate(row) if val]
        if connections:
            description += f"Node {i + 1} is connected to nodes {', '.join(connections)}.\n"
        else:
            description += f"Node {i + 1} has no connections.\n"
    return description

def paths_to_text(paths):
    """
    Convert a list of paths to a textual description.
    """
    description = ""
    for idx, path in enumerate(paths):
        description += f"Path {idx + 1}: {' -> '.join(map(str, path))}\n"
    return description

def generate_data_description(matrix_id, size, adjacency_matrix, start_node, end_node):
    """
    Generate a textual or JSON description of the adjacency matrix based on settings.
    """
    if not adjacency_matrix:
        print(f"Matrix ID {matrix_id} has inconsistent data.")
        return None

    if USE_TEXT_DESCRIPTION:
        adjacency_text = adjacency_matrix_to_text(adjacency_matrix)
        return f"network description:\n{adjacency_text}"
    else:
        matrix_data = {
            "matrix_id": matrix_id,
            "size": size,
            "distances_enabled": False,
            "adjacency_matrix": adjacency_matrix
        }
        return f"adjacency matrix:\n{json.dumps(matrix_data, indent=4)}"

def call_gpt_api(prompt):
    """
    Call the GPT API with the given prompt and return the parsed JSON response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        gpt_message = response.choices[0].message.content.strip()
        if not (gpt_message.startswith('{') and gpt_message.endswith('}')):
            raise ValueError("GPT response is not in the expected JSON format.")
        return json.loads(gpt_message)
    except Exception as e:
        print(f"Error communicating with GPT: {e}")
        return None

def get_gpt_all_connections(matrix_id, size, adjacency_matrix=None, start_node=1, end_node=None):
    """
    Send a prompt to GPT to list all possible paths between start and end nodes.
    """
    if end_node is None:
        end_node = size  # Default to the last node

    data_description = generate_data_description(matrix_id, size, adjacency_matrix, start_node, end_node)
    if data_description is None:
        return None

    prompt = f"""
        Given the following {data_description}

        Please list all possible paths from node {start_node} to node {end_node} based on the adjacency information.

        **Important:** Only provide the result in the following JSON format without any additional text or explanations. Ensure that the response contains **only** the JSON.

        {{
            "matrix_id": {matrix_id},
            "all_paths": [
                [node_sequence_1],
                [node_sequence_2],
                ...
            ]
        }}
    """
    gpt_response = call_gpt_api(prompt)
    return gpt_response.get('all_paths', []) if gpt_response else None

def get_gpt_choose_shortest_path(matrix_id, all_paths):
    """
    Send a prompt to GPT to choose the shortest path from a list of paths.
    """
    if USE_TEXT_DESCRIPTION:
        paths_info = f"list of paths:\n{paths_to_text(all_paths)}"
    else:
        paths_info = f"list of paths:\n{json.dumps(all_paths, indent=4)}"

    prompt = f"""
        Given the following {paths_info}

        Please identify the shortest path from the above list. If multiple paths have the same shortest length, you may choose any one of them.

        Note that the connection part in shortest_distance is the number of connected lines, that is, the number of nodes minus 1.

        **Important:** Only provide the result in the following JSON format without any additional text or explanations. Ensure that the response contains **only** the JSON.

        {{
            "matrix_id": {matrix_id},
            "shortest_path": [node_sequence],
            "shortest_distance": total_connections
        }}
    """
    return call_gpt_api(prompt)

def get_gpt_shortest_path(matrix_id, size, adjacency_matrix=None, start_node=1, end_node=None):
    """
    Send a prompt to GPT to directly return the shortest path and its distance.
    """
    if end_node is None:
        end_node = size  # Default to the last node

    data_description = generate_data_description(matrix_id, size, adjacency_matrix, start_node, end_node)
    if data_description is None:
        return None

    prompt = f"""
        Given the following {data_description}

        Please identify the shortest path from node {start_node} to node {end_node} based on the adjacency information.

        **Important:** Only provide the result in the following JSON format without any additional text or explanations. Ensure that the response contains **only** the JSON.

        {{
            "matrix_id": {matrix_id},
            "shortest_path": [node_sequence],
            "shortest_distance": total_connections
        }}
    """
    return call_gpt_api(prompt)

def main():

    filename = "adjacency_matrices.json"
    matrices = read_json(filename)

    if not matrices:
        print("Failed to read data. Please ensure the file exists and is correctly formatted.")
        return

    results = []
    correct_count = 0
    total_count = len(matrices)
    start_time = time.time()

    for matrix in tqdm(matrices, desc="Processing Matrices", unit="matrix"):
        matrix_id = matrix.get('matrix_id', 'Unknown')
        size = matrix.get('size')
        adjacency_matrix = matrix.get('adjacency_matrix')
        if not adjacency_matrix:
            print(f"Matrix ID {matrix_id} is missing 'adjacency_matrix'.")
            continue

        start_node = 1
        end_node = size
        calculated_path, calculated_distance = bfs_shortest_path(adjacency_matrix, start_node - 1, end_node - 1)

        result = {
            "matrix_id": matrix_id,
            "calculated_path": calculated_path,
            "calculated_distance": calculated_distance,
            "gpt_path": [],
            "gpt_distance": -1,
            "is_correct": False
        }

        if TWO_STEP_PROCESS:
            gpt_all_connections = get_gpt_all_connections(
                matrix_id, size, adjacency_matrix, start_node, end_node
            )
            if not gpt_all_connections:
                print(f"No connections found by GPT for Matrix ID {matrix_id}.")
                results.append(result)
                continue

            gpt_response = get_gpt_choose_shortest_path(matrix_id, gpt_all_connections)
            if not gpt_response:
                print(f"GPT failed to choose the shortest path for Matrix ID {matrix_id}.")
                results.append(result)
                continue
        else:
            gpt_response = get_gpt_shortest_path(
                matrix_id, size, adjacency_matrix, start_node, end_node
            )
            if not gpt_response:
                print(f"GPT failed to provide the shortest path for Matrix ID {matrix_id}.")
                results.append(result)
                continue

        gpt_path = gpt_response.get('shortest_path', [])
        gpt_distance = gpt_response.get('shortest_distance', -1)
        result.update({
            "gpt_path": gpt_path,
            "gpt_distance": gpt_distance,
            "is_correct": calculated_distance == gpt_distance
        })
        if result["is_correct"]:
            correct_count += 1

        results.append(result)

    end_time = time.time()
    total_time = end_time - start_time
    accuracy = (correct_count / total_count) * 100 if total_count else 0

    output = {
        "results": results,
        "GPT_accuracy": f"{correct_count} out of {total_count} correct ({accuracy:.2f}%)",
        "Total_execution_time": f"{total_time:.2f} seconds"
    }

    output_filename = "verification_results.json"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print(f"\nVerification results saved to '{output_filename}'.")
    except IOError as e:
        print(f"Error saving results to file: {e}")

    print(f"\nGPT's Accuracy: {correct_count} out of {total_count} correct ({accuracy:.2f}%)")
    print(f"Total Execution Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
