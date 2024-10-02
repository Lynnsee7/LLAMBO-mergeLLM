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
    metric_label = 'Avg Regret'
    is_maximization = True

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


                if is_maximization:
                    global_best_value = max(global_best_value, max(current_metrics))
                    global_worst_value = min(global_worst_value, min(current_metrics))
                else:
                    global_best_value = min(global_best_value, min(current_metrics))
                    global_worst_value = max(global_worst_value, max(current_metrics))

            avg_metrics = [sum(x) / len(x) for x in zip(*experiment_metrics)]
            all_metrics.append((optimizer_name, avg_metrics))


    if all_metrics:
        for optimizer_name, avg_metrics in all_metrics:
            if len(avg_metrics) > 30:
                avg_metrics = avg_metrics[:30]

            current_max = float('-inf') if is_maximization else float('inf')
            regret_values = []

            for value in avg_metrics:
                if is_maximization:
                    current_max = max(current_max, value)

                regret = (current_max - global_best_value) / (global_worst_value - global_best_value)
                regret_values.append(regret)


            if regret_values:
                axes[plot_index].plot(range(1, len(regret_values) + 1), regret_values, label=optimizer_name)


        axes[plot_index].set_xlabel('Iterations')
        axes[plot_index].set_ylabel(metric_label)
        axes[plot_index].set_title(f'{model} {dataset}')
        axes[plot_index].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        axes[plot_index].grid(True)

        plot_index += 1

plt.tight_layout()


combined_svg_filename = '/root/LLAMBO/results_imgs/hpo_individual_regret/hpo_combined_regret.svg'
plt.savefig(combined_svg_filename, format='svg', bbox_inches='tight')

print("Combined regret plots generated successfully.")
