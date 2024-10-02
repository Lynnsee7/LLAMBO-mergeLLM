# # draw separate pictures
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
#                 metric_label = 'Avg Regret'
#                 is_maximization = True
#             else:
#                 metric_key = 'mean_absolute_error'
#                 metric_label = 'Avg Regret'
#                 is_maximization = False
            
#             all_metrics = []
#             global_best_value = float('-inf') if is_maximization else float('inf')
#             global_worst_value = float('inf') if is_maximization else float('-inf')

#             for optimizer_folder in os.listdir(task_path):
#                 optimizer_path = os.path.join(task_path, optimizer_folder)
#                 optimizer_name = optimizer_folder.split(' ')[0]
                
#                 metrics_file = os.path.join(optimizer_path, 'metrics.json')
#                 if os.path.exists(metrics_file):
#                     with open(metrics_file, 'r') as f:
#                         metrics = json.load(f)
                    
#                     experiment_metrics = []
#                     for experiment in metrics:
#                         current_metrics = [iteration[metric_key] for iteration in experiment]
#                         experiment_metrics.append(current_metrics)

#                         # 更新全局的 best 和 worst 值
#                         if is_maximization:
#                             global_best_value = max(global_best_value, max(current_metrics))
#                             global_worst_value = min(global_worst_value, min(current_metrics))
#                         else:
#                             global_best_value = min(global_best_value, min(current_metrics))
#                             global_worst_value = max(global_worst_value, max(current_metrics))
                    
#                     avg_metrics = [sum(x) / len(x) for x in zip(*experiment_metrics)]
#                     all_metrics.append((optimizer_name, avg_metrics))
            
#             for optimizer_name, avg_metrics in all_metrics:
#                 if len(avg_metrics) > 30:
#                     avg_metrics = avg_metrics[:30]
                
#                 current_max = float('-inf') if is_maximization else float('inf')
#                 regret_values = []

#                 for value in avg_metrics:

#                     if is_maximization:
#                         current_max = max(current_max, value)
#                     else:
#                         current_max = min(current_max, value)


#                     regret = (current_max - global_best_value) / (global_worst_value - global_best_value)
#                     regret_values.append(regret)
                
#                 print(f"Dataset: {dataset}, Model: {model}, Optimizer: {optimizer_name}, Length of regret_values: {len(regret_values)}")
#                 plt.plot(range(1, len(regret_values) + 1), regret_values, label=optimizer_name)

#             plt.xlabel('Iterations')
#             plt.ylabel(metric_label)
#             plt.title(f'{model} {dataset}')
#             plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
#             plt.grid(True)
            
#             svg_filename = f'/root/LLAMBO/results_imgs/bayesmark_individual_regret/{task_type}_{dataset}_{model}_regret.svg'
#             plt.savefig(svg_filename, format='svg', bbox_inches='tight')
#             plt.clf()

# print("Regret plots generated successfully.")


import os
import json
import matplotlib.pyplot as plt
import math

results_folder = '/root/LLAMBO/results'

task_types = ['metric_acc_data', 'metric_mse_data']

# Count the number of datasets and models to dynamically calculate the grid size for subplots
total_plots = 0

for task_type in task_types:
    for task_folder in os.listdir(results_folder):
        if task_folder.startswith(task_type):
            total_plots += 1

# Determine the grid size for subplots based on the total number of plots (e.g., 2 rows)
cols = 4  # You can adjust the number of columns based on your preference
rows = math.ceil(total_plots / cols)

# Create a figure with subplots
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
axes = axes.flatten()  # Flatten the axes array to iterate over it easily

plot_index = 0  # To track the current subplot

for task_type in task_types:
    for task_folder in os.listdir(results_folder):
        if task_folder.startswith(task_type):
            task_path = os.path.join(results_folder, task_folder)
            
            dataset = task_folder.split('_')[3]
            model = task_folder.split('_')[5]

            if task_type == 'metric_acc_data':
                metric_key = 'accuracy'
                metric_label = 'Avg Regret'
                is_maximization = True
            else:
                metric_key = 'mean_absolute_error'
                metric_label = 'Avg Regret'
                is_maximization = False
            
            all_metrics = []
            global_best_value = float('-inf') if is_maximization else float('inf')
            global_worst_value = float('inf') if is_maximization else float('-inf')

            for optimizer_folder in os.listdir(task_path):
                optimizer_path = os.path.join(task_path, optimizer_folder)
                optimizer_name = optimizer_folder.split(' ')[0]
                
                metrics_file = os.path.join(optimizer_path, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    experiment_metrics = []
                    for experiment in metrics:
                        current_metrics = [iteration[metric_key] for iteration in experiment]
                        experiment_metrics.append(current_metrics)

                        # 更新全局的 best 和 worst 值
                        if is_maximization:
                            global_best_value = max(global_best_value, max(current_metrics))
                            global_worst_value = min(global_worst_value, min(current_metrics))
                        else:
                            global_best_value = min(global_best_value, min(current_metrics))
                            global_worst_value = max(global_worst_value, max(current_metrics))
                    
                    avg_metrics = [sum(x) / len(x) for x in zip(*experiment_metrics)]
                    all_metrics.append((optimizer_name, avg_metrics))
            
            for optimizer_name, avg_metrics in all_metrics:
                if len(avg_metrics) > 30:
                    avg_metrics = avg_metrics[:30]
                
                current_max = float('-inf') if is_maximization else float('inf')
                regret_values = []

                for value in avg_metrics:

                    if is_maximization:
                        current_max = max(current_max, value)
                    else:
                        current_max = min(current_max, value)

                    regret = (current_max - global_best_value) / (global_worst_value - global_best_value)
                    regret_values.append(regret)
                
                # Plot on the current subplot
                axes[plot_index].plot(range(1, len(regret_values) + 1), regret_values, label=optimizer_name)

            # Set labels and titles for each subplot
            axes[plot_index].set_xlabel('Iterations')
            axes[plot_index].set_ylabel(metric_label)
            axes[plot_index].set_title(f'{model} {dataset}')
            axes[plot_index].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
            axes[plot_index].grid(True)

            plot_index += 1

# Adjust layout to avoid overlapping
plt.tight_layout()

# Save the combined figure as SVG
combined_svg_filename = '/root/LLAMBO/results_imgs/bayesmark_individual_regret/combined_regret_plots.svg'
plt.savefig(combined_svg_filename, format='svg', bbox_inches='tight')

print("Combined regret plots generated successfully.")
