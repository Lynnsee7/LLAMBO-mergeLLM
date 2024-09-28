import pickle
import json
import os
import pandas as pd
import numpy as np

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False

def convert_to_json_serializable(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def pickle_to_json(pickle_file_path, json_file_path):
    print("1111111111")  # 调试信息

    if not os.path.exists(pickle_file_path):
        print(f"Error: The file {pickle_file_path} does not exist.")
        return

    try:
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Successfully read {pickle_file_path}")  # 调试信息
    except Exception as e:
        print(f"Error reading {pickle_file_path}: {e}")
        return

    try:
        # 递归地将数据转换为 JSON 可序列化的格式
        data = convert_to_json_serializable(data)
    except TypeError as e:
        print(f"Error: The data structure in {pickle_file_path} is not JSON serializable.")
        print(f"Details: {e}")
        return

    try:
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
            print(f"Successfully wrote {json_file_path}")  # 调试信息
    except Exception as e:
        print(f"Error writing {json_file_path}: {e}")

def process_directory(directory):
    print(f"Processing directory: {directory}")  # 调试信息
    for root, dirs, files in os.walk(directory):
        print(f"Current directory: {root}")  # 调试信息
        for file in files:
            if file.endswith('.pickle'):  # 注意这里检查的是 .pickle 后缀
                pickle_file_path = os.path.join(root, file)
                json_file_path = os.path.join(root, file.replace('.pickle', '.json'))
                print(f"Processing file: {pickle_file_path}")  # 调试信息
                pickle_to_json(pickle_file_path, json_file_path)

if __name__ == "__main__":
    # 输入文件夹路径
    directory_path = '/root/LLAMBO/results'  # 替换为你的文件夹路径

    # 检查文件夹路径是否存在
    if not os.path.exists(directory_path):
        print(f"Error: The directory {directory_path} does not exist.")
    else:
        # 处理文件夹中的所有pickle文件
        process_directory(directory_path)