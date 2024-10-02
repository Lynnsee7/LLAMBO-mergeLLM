import os
import json
import matplotlib.pyplot as plt
import math

results_folder = '/root/LLAMBO/results'
task_types = ['metric_acc_data', 'metric_mse_data']


optimizer_regret_dict = {}


for task_type in task_types:
    for task_folder in os.listdir(results_folder):
        if task_folder.startswith(task_type):
            task_path = os.path.join(results_folder, task_folder)
            
            dataset = task_folder.split('_')[3]
            model = task_folder.split('_')[5]

            if task_type == 'metric_acc_data':
                metric_key = 'accuracy'
                is_maximization = True
            else:
                metric_key = 'mean_absolute_error'
                is_maximization = False
            
            global_best_value = float('-inf') if is_maximization else float('inf')
            global_worst_value = float('inf') if is_maximization else float('-inf')

            
            task_optimizer_metrics = {}

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
                    
                    
                    if optimizer_name not in task_optimizer_metrics:
                        task_optimizer_metrics[optimizer_name] = []

                    task_optimizer_metrics[optimizer_name].append(avg_metrics)


            for optimizer_name, avg_metrics_list in task_optimizer_metrics.items():
                current_max = float('-inf') if is_maximization else float('inf')
                regret_values = []

                for avg_metrics in avg_metrics_list[0]:  
                    if is_maximization:
                        current_max = max(current_max, avg_metrics)
                    else:
                        current_max = min(current_max, avg_metrics)

                    regret = (current_max - global_best_value) / (global_worst_value - global_best_value)
                    regret = max(0, min(1, regret)) 
                    regret_values.append(regret)


                if optimizer_name not in optimizer_regret_dict:
                    optimizer_regret_dict[optimizer_name] = []
                
                optimizer_regret_dict[optimizer_name].append(regret_values)


optimizer_avg_regret = {}

for optimizer_name, all_task_regrets in optimizer_regret_dict.items():
    total_tasks = len(all_task_regrets)

    avg_regret_per_iteration = [sum(x) / total_tasks for x in zip(*all_task_regrets)]

    optimizer_avg_regret[optimizer_name] = avg_regret_per_iteration


plt.figure(figsize=(10, 6))


for optimizer_name, avg_regret in optimizer_avg_regret.items():
    if len(avg_regret) > 30:
        avg_regret = avg_regret[:30]  

    plt.plot(range(1, len(avg_regret) + 1), avg_regret, label=optimizer_name)


plt.xlabel('Iterations')
plt.ylabel('Avg Regret')
plt.title('Average Regret Across All Tasks')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.grid(True)


combined_svg_filename = '/root/LLAMBO/results_imgs/all_tasks/bayesmark_avg_regret_across_tasks.svg'
plt.savefig(combined_svg_filename, format='svg', bbox_inches='tight')

print("Average regret plot across all tasks generated successfully.")
