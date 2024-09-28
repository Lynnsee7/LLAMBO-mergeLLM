import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

results_dir = '/root/LLAMBO/results'


n_trials_per_experiment = 30
n_experiments = 5 


method_results = {}

for metric_folder in os.listdir(results_dir):
    metric_path = os.path.join(results_dir, metric_folder)
    
    if os.path.isdir(metric_path):
        for method_folder in os.listdir(metric_path):
            method_path = os.path.join(metric_path, method_folder)

            if os.path.isdir(method_path) and "(Random init)" in method_folder:
                method_name = method_folder.split(" (Random init)")[0]
                

                if method_name not in method_results:
                    method_results[method_name] = []


                metrics_file = os.path.join(method_path, "metrics.pickle")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'rb') as f:
                        metrics = pickle.load(f)


                    accuracy_values = []
                    for entry in metrics:
                        if 'accuracy' in entry:
                            accuracy_values.append(entry['accuracy'])

                    if len(accuracy_values) > 0:
                        results = np.array(accuracy_values)
                    

                        results = results.reshape(n_experiments, n_trials_per_experiment)

                        avg_regrets_per_experiment = np.cumsum(results, axis=1) / np.arange(1, n_trials_per_experiment + 1)
                        

                        method_results[method_name].append(np.mean(avg_regrets_per_experiment, axis=0))


for method_name, regrets in method_results.items():
    method_results[method_name] = np.mean(regrets, axis=0)

plt.figure(figsize=(8, 6))
for method_name, avg_regret in method_results.items():
    plt.plot(range(1, len(avg_regret) + 1), avg_regret, label=method_name)

plt.xlabel('Trials')
plt.ylabel('Avg Regret')
plt.title('Baselines')
plt.legend()
plt.grid(False) 


plt.savefig(f"baselines_regclf.png")
plt.show()
