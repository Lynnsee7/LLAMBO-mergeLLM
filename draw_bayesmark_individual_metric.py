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

# # draw all in one picture
import os
import json
import matplotlib.pyplot as plt
import math

results_folder = '/root/LLAMBO/results'
task_types = ['metric_acc_data', 'metric_mse_data']

# 计算需要生成的图的总数，方便设置合适的子图网格
total_plots = 0

for task_type in task_types:
    for task_folder in os.listdir(results_folder):
        if task_folder.startswith(task_type):
            total_plots += 1

# 动态确定网格的大小（行和列）
cols = 4  # 每行显示3个图，你可以根据需要调整
rows = math.ceil(total_plots / cols)

# 创建一个包含所有子图的画布
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
axes = axes.flatten()  # 将子图数组展平，方便迭代处理

plot_index = 0  # 用于跟踪当前处理的子图

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

            for optimizer_name, avg_metrics in all_metrics:
                if len(avg_metrics) > 30:
                    avg_metrics = avg_metrics[:30]
                
                max_values = []
                current_max = float('-inf')
                for value in avg_metrics:
                    if value > current_max:
                        current_max = value
                    max_values.append(current_max)
                
                print(f"Dataset: {dataset}, Model: {model}, Optimizer: {optimizer_name}, Length of avg_metrics: {len(avg_metrics)}")
                # 在当前的子图上作图
                axes[plot_index].plot(range(1, len(max_values) + 1), max_values, label=optimizer_name)

            # 设置当前子图的标题和标签
            axes[plot_index].set_xlabel('Iterations')
            axes[plot_index].set_ylabel(metric_label)
            axes[plot_index].set_title(f'{model} {dataset}')
            axes[plot_index].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
            axes[plot_index].grid(True)

            plot_index += 1

# 调整布局以避免重叠
plt.tight_layout()

# 保存合并后的图像
combined_svg_filename = '/root/LLAMBO/results_imgs/combined_results.svg'
plt.savefig(combined_svg_filename, format='svg', bbox_inches='tight')

print("拼接SVG已生成")
