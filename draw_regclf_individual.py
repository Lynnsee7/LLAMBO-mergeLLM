# separate drawing
# import os
# import json
# import matplotlib.pyplot as plt
# import svgwrite

# results_folder = '/root/LLAMBO/results'

# task_types = ['metric_acc_data', 'metric_mse_data']


# plt.figure(figsize=(10, 6))


# for task_type in task_types:
#     for task_folder in os.listdir(results_folder):
#         if task_folder.startswith(task_type):
#             task_path = os.path.join(results_folder, task_folder)
            
#             dataset = task_folder.split('_')[3]
#             model = task_folder.split('_')[5]

#             if task_type == 'metric_acc_data':
#                 metric_key = 'accuracy'
#                 metric_label = 'Accuracy'
#             else:
#                 metric_key = 'mean_absolute_error'
#                 metric_label = 'MSE'
            
#             all_metrics = []
            

#             for optimizer_folder in os.listdir(task_path):
#                 optimizer_path = os.path.join(task_path, optimizer_folder)
                

#                 optimizer_name = optimizer_folder.split(' ')[0] 
                

#                 metrics_file = os.path.join(optimizer_path, 'metrics.json')
#                 if os.path.exists(metrics_file):
#                     with open(metrics_file, 'r') as f:
#                         metrics = json.load(f)
                  
#                     experiment_metrics = []
#                     for experiment in metrics:
#                         if task_type == 'metric_acc_data':
#                             experiment_metrics.append([iteration[metric_key] for iteration in experiment])
#                         else:
#                             experiment_metrics.append([-iteration[metric_key] for iteration in experiment])
                    
#                     avg_metrics = [sum(x) / len(x) for x in zip(*experiment_metrics)]
#                     all_metrics.append((optimizer_name, avg_metrics))
            
#             for optimizer_name, avg_metrics in all_metrics:
#                 # for smac
#                 if len(avg_metrics) > 30:
#                     avg_metrics = avg_metrics[:30]
#                 max_values = []
#                 current_max = float('-inf')
#                 for value in avg_metrics:
#                     if value > current_max:
#                         current_max = value
#                     max_values.append(current_max)
                
#                 print(f"Dataset: {dataset}, Model: {model}, Optimizer: {optimizer_name}, Length of avg_metrics: {len(avg_metrics)}")
#                 plt.plot(range(1, len(max_values) + 1), max_values, label=optimizer_name)
                    

#             plt.xlabel('Iterations')
#             plt.ylabel(metric_label)
#             plt.title(f'{model} {dataset}')
#             # plt.legend()
#             plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
#             plt.grid(True)
            
#             svg_filename = f'/root/LLAMBO/results_imgs/{task_type}_{dataset}_{model}.svg'
#             plt.savefig(svg_filename, format='svg', bbox_inches='tight')
#             plt.clf()

# print("SVG OK")

# draw all in one picture
import os
import json
import matplotlib.pyplot as plt
from PIL import Image

results_folder = '/root/LLAMBO/results'

task_types = ['metric_acc_data', 'metric_mse_data']

# 定义临时文件夹路径
temp_folder = '/root/LLAMBO/temp_imgs'
os.makedirs(temp_folder, exist_ok=True)

# 定义最终拼接图片的保存路径
final_image_path = '/root/LLAMBO/results_imgs/combined_results.png'

# 遍历任务类型
for task_type in task_types:
    for task_folder in os.listdir(results_folder):
        if task_folder.startswith(task_type):
            task_path = os.path.join(results_folder, task_folder)
            
            dataset = task_folder.split('_')[3]
            model = task_folder.split('_')[5]

            if task_type == 'metric_acc_data':
                metric_key = 'accuracy'
                metric_label = 'Accuracy'
            else:
                metric_key = 'mean_absolute_error'
                metric_label = 'MSE'
            
            all_metrics = []
            
            for optimizer_folder in os.listdir(task_path):
                optimizer_path = os.path.join(task_path, optimizer_folder)
                
                optimizer_name = optimizer_folder.split(' ')[0] 
                
                metrics_file = os.path.join(optimizer_path, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                  
                    experiment_metrics = []
                    for experiment in metrics:
                        if task_type == 'metric_acc_data':
                            experiment_metrics.append([iteration[metric_key] for iteration in experiment])
                        else:
                            experiment_metrics.append([-iteration[metric_key] for iteration in experiment])
                    
                    avg_metrics = [sum(x) / len(x) for x in zip(*experiment_metrics)]
                    all_metrics.append((optimizer_name, avg_metrics))
            
            plt.figure(figsize=(10, 6))
            for optimizer_name, avg_metrics in all_metrics:
                if len(avg_metrics) > 30:
                    avg_metrics = avg_metrics[:30]
                
                # 计算当前最大值
                max_values = []
                current_max = float('-inf')
                for value in avg_metrics:
                    if value > current_max:
                        current_max = value
                    max_values.append(current_max)
                
                print(f"Dataset: {dataset}, Model: {model}, Optimizer: {optimizer_name}, Length of avg_metrics: {len(avg_metrics)}")
                plt.plot(range(1, len(max_values) + 1), max_values, label=optimizer_name)

            plt.xlabel('Iterations')
            plt.ylabel(metric_label)
            plt.title(f'{model} {dataset}')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
            plt.grid(False)
            
            temp_image_path = os.path.join(temp_folder, f'{task_type}_{dataset}_{model}.png')
            plt.savefig(temp_image_path, format='png', bbox_inches='tight')
            plt.clf()

# 拼接图片
images = [Image.open(os.path.join(temp_folder, f)) for f in os.listdir(temp_folder) if os.path.isfile(os.path.join(temp_folder, f))]
widths, heights = zip(*(i.size for i in images))

max_width = max(widths)
total_height = sum(heights)

new_image = Image.new('RGB', (max_width, total_height))

y_offset = 0
for image in images:
    # 调整图片宽度
    if image.size[0] < max_width:
        new_image.paste(image, (0, y_offset))
    else:
        new_image.paste(image, (0, y_offset))
    y_offset += image.size[1]

new_image.save(final_image_path)

print("拼接图片已生成")
