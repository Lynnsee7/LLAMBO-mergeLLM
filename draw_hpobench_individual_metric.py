import os
import json
import matplotlib.pyplot as plt
import math

results_folder = '/root/LLAMBO/results_for_draw/hpo'


task_folders = [folder for folder in os.listdir(results_folder) 
                if os.path.isdir(os.path.join(results_folder, folder)) and 'checkpoints' not in folder]

total_plots = len(task_folders)  
cols = 4  
rows = math.ceil(total_plots / cols)


fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
axes = axes.flatten()  

plot_index = 0 

for task_folder in task_folders: 
    task_path = os.path.join(results_folder, task_folder)

    dataset = task_folder.split('_')[0]
    model = task_folder.split('_')[1]

    metric_key = 'acc'
    metric_label = 'Accuracy'

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
                experiment_metrics.append([iteration[metric_key] for iteration in experiment])

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
     
        axes[plot_index].plot(range(1, len(max_values) + 1), max_values, label=optimizer_name)

  
    axes[plot_index].set_xlabel('Iterations')
    axes[plot_index].set_ylabel(metric_label)
    axes[plot_index].set_title(f'{model} {dataset}')
    axes[plot_index].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    axes[plot_index].grid(True)

    plot_index += 1


plt.tight_layout()


combined_svg_filename = '/root/LLAMBO/results_imgs/hpo_individual_metric/combined_hpo_metric.svg'
plt.savefig(combined_svg_filename, format='svg', bbox_inches='tight')

print("拼接SVG已生成")
