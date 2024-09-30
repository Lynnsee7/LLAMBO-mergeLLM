import re
import time
import copy
import torch
import random
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import ConfigSpace as CS
from transformers import AutoTokenizer, AutoModelForCausalLM
from sampler import obtain_space
import optuna
from optuna.samplers import *
from optuna.study._optimize import _run_trial
from bo_models.bo_tpe import bo_tpe



class bops_model_optuna:
    def __init__(self,model_path,prompt_type,temperature=0.6):
        # create agent
        self.model_path=model_path
        self.prompt_type=prompt_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,device_map='auto')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        self.temperature=temperature
        # print('init done!')
        
    def generate_response(self,prompt, max_new_tokens=128):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs.input_ids, max_new_tokens=max_new_tokens,temperature=self.temperature,do_sample=True)
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response

    def _call_model(self, task_config, history, e,dup):
        print("now calling model=============================")
        # Prompt for LLM
        self.data = "You are a powerful Bayesian optimizer capable of wisely deciding the best combination of optimization parameters based on the historical information from Bayesian optimization."
        # valid parameters for skopt
        
        allowed_params = 'Sampler: GPSampler,TPESampler,CmaEsSampler,RandomSampler,NSGAIISampler,NSGAIIISampler,QMCSampler'
        if self.prompt_type=='detail':
            allowed_params+=''
        
        # initialize history data
        target_score = 'a fairly good performance'
        history_configs=['Performance:fairly good.\n Hyperparameter configuration:']
        if history['records'] == []:
            # target score
            target_score = 'a fairly good performance'

            # history
            history_str = ''
        # if the history data is not empty, update it.
        else:
            # target score:minimize,here we assume the expected performance is a 10% gain.
            if history['best_score'] > 0:
                target_score = history['best_score'] * 0.9
            elif history['best_score'] < 0:
                target_score = history['best_score'] * 1.1

            # remain the latest five history entry
            if len(history['records']) > 5:
                history_cus = history['records'][-5:-1]
            else:
                history_cus=history['records']
            
            # load history data to history_configs
            for i in range(len(history_cus)):
                record = history_cus[i]
                entry = f"--Step {i}-- Performance: {record['performance']}\nHyperparameter configuration: ## Sampler:{record['smpl']}; ##"
                history_configs.append(entry)
            history_configs.append(
                f"--Step {len(history_cus)}-- Performance: {target_score}\nHyperparameter configuration:")
            history_str = '\n'.join(history_configs)
        
        if self.prompt_type=='detail':
            prompt_user=f"""
            \nThe following are trajectories of the performance of a {task_config['model']} measured in {task_config['metric']} and the corresponding bayesian optimizer hyperparameter configurations. The model is evaluated on {task_config['task']}Here's a description of the dataset:{task_config['desc']}\n\nThe allowable ranges for the optimizer hyperparameters are: {allowed_params}. Recommend a configuration(one surrogate model,one acquisition function, and one acquisition optimizer ) that can achieve the target performance of {target_score}. Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. Recommend values with the highest possible precision, as requested by the allowed ranges.  """
        else:    
            prompt_user = f"""
        The following are examples of the performance of a {task_config['model']} measured in {task_config['metric']} and the corresponding bayesian optimizer hyperparameter configurations. The model is evaluated on {task_config['task']}.The allowable ranges for the optimizer hyperparameters are: {allowed_params}. Recommend a configuration(one Sampler) that can achieve the target performance of {target_score}. Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. Recommend values with the highest possible precision, as requested by the allowed ranges."""
        # handle errors
        
        if e is not None:
            error_information = f"when selecting params,please avoid this error:{e}\n"
            prompt_user += error_information
        prompt_user+=" Your response is in one line in the format '## Sampler: ...; ##'.\n "
        
        
        self.data+=prompt_user
        # print(prompt_user)
        # form prompt
        if dup is not None:
            # avoid duplication(when making dpo dataset)
            dup_information=f"When selecting params,be different from this combination:{dup}\n"
            # print(dup_information)
            self.data+=dup_information
        if self.prompt_type!='no_history':
            self.data+=history_str
        else:
            self.data+=history_configs[-1]
        # print(prompt_user)
        # form prompt
        print('data:',self.data)
        original_response = self.generate_response(self.data)
        # print('resp:',original_response)
        # refine response
        responses = [item.strip() for item in original_response.strip().split('\n')if len(item.strip())>0][::-1]
        # print(responses)
        response=None
        for item in responses:
            if item[0]=='#':
                response=item
                break
        if response==None:
            response=' '.join(responses)
            
        # print('resp:',response)
        try:
            return response
        except Exception as e:
            print('content generation failed with ',e)

    def _parse_params(self, resp):
        match = re.search(
            r"Sampler:\s*(.*?);", resp)
        # print('match:',match)
        if match:
            smpl = match.group(1).strip()
            return smpl
        else:
            print('resp of llm:  ', resp)

    def __call__(self, task_config, history, e,dup):
        # start_time = time.time()
        resp = self._call_model(task_config, history, e,dup)
        # end_time = time.time()

        # execution_time = end_time - start_time
        # print(f"Execution time for _call_model: {execution_time:.4f} seconds")

        smpl = self._parse_params(resp)
        return smpl

class OptunaExpLLMOptimizer:
    def __init__(self,
                 objective_function,
                 search_spaces,
                 task_config,
                 task_config_simple,
                 model_path='/root/autodl-tmp/vicuna',
                 prompt_type=None,
                 #  bops_model_skopt,
                 #  base_estimator: str = 'GP',
                 #  acq_func: str = 'EI',
                 #  acq_optimizer: str = 'auto',
                 direction="minimize"
                 ):
        print("initialize OptunaLLMOptimizer")
        self.objective = objective_function
        self.search_spaces=search_spaces
        self.model_path=model_path
        self.prompt_type=prompt_type
        self.model = bops_model_optuna(model_path,self.prompt_type)
        self.direction = direction
        self.optimizer=None
        self.history = {
            'records': [],
            'best_score': 0
        }
        self.task_config = task_config  # used for prompting the model
        self.task_config_simple = task_config_simple  # used for formatting save paths
        self.scores_history = None
        self.params_history = None
        self.prompt_type=prompt_type
        self.sampler_map = {
    'TPESampler': TPESampler(),
    'GPSampler': GPSampler(),
    'CmaEsSampler': CmaEsSampler(),
    # 'GridSampler': GridSampler(),
    'RandomSampler': RandomSampler(),
    # 'PartialFixedSampler': PartialFixedSampler(),
    'NSGAIISampler': NSGAIISampler(),
    'NSGAIIISampler': NSGAIIISampler(),
    'QMCSampler': QMCSampler()
}

    def _param_update(self, task_config, history, e=None,dup=None):
        while True:
            try:
                smpl= self.model(task_config, history, e,dup)
                smpl = self._validate_base_sampler(smpl)
        
                    
        
                print(f'selected sampler:{smpl}')
                return smpl
            except ValueError as err:
                e=err
                print('value error:',e)
            
    def _validate_base_sampler(self, base_sampler):
        valid_samplers = ['GPSampler', 'TPESampler', 'CmaEsSampler','GridSampler', 'RandomSampler', 
                          'PartialFixedSampler',
                          'NSGAIISampler', 'NSGAIIISampler', 'QMCSampler']
        if base_sampler not in valid_samplers:
            raise ValueError(f"Invalid base_estimator '{base_sampler}'. Must be one of {valid_samplers}.")
        return base_sampler


    def create_optimizer(self, params):
        """
        Creates an optimizer instance for a given search space and configuration.
        """
        optimizer = optuna.create_study(sampler=self.sampler_map[params],
                            pruner= None,
                            direction=self.direction) # TODO check if it fits optuna
        return optimizer

    @staticmethod
    def copy_attributes(source, target):
        for attr, value in vars(source).items():
            setattr(target, attr, value)

    def save_history(self, save_path='./'):
        if self.params_history == None or self.scores_history == None:
            raise ValueError('the optimizer has no optimization history!')

        params_array = np.array(self.params_history)
        scores_array = np.array(self.scores_history)
        cumulative_min_scores = np.minimum.accumulate(scores_array)
        points = [(i, score) for i, score in enumerate(cumulative_min_scores)]

        # Write the points to 'skopt.txt'
        if 'dpo' in self.model_path.lower():
            save_file=os.path.join(save_path,f"vicunaoptunadpo_{self.task_config_simple['task']}_{self.task_config_simple['model']}.txt")
            # Write the points to 'skopt.txt': ablation1-ensure using the best tuned model.
            if self.prompt_type=='detail':
                save_file=os.path.join(save_path,f"vicunaoptunadpodetail_{self.task_config_simple['task']}_{self.task_config_simple['model']}.txt")
            elif self.prompt_type=='no_history':
                save_file=os.path.join(save_path,f"vicunaoptunadponohistory_{self.task_config_simple['task']}_{self.task_config_simple['model']}.txt")
                
        elif 'sft' in self.model_path.lower():
            save_file=os.path.join(save_path,f"vicunaoptunasft_{self.task_config_simple['task']}_{self.task_config_simple['model']}.txt")
        elif 'vicuna' in self.model_path.lower():
            save_file=os.path.join(save_path,f"vicunaoptunaoriginal_{self.task_config_simple['task']}_{self.task_config_simple['model']}.txt")
        with open(save_file, 'w') as f:
            for x, y in points:
                f.write(f"{x},{y}\n")

    def plot_optimization_results(self, params_history, scores_history, filename):
        """
        Creates and saves plots showing the optimization results, and writes the (x, y) points to 'skopt.txt'.

        Parameters:
        - params_history: List of parameter sets from optimization.
        - scores_history: List of objective function values.
        - filename: The name of the file to save the plots.
        """
        scores_array = np.array(scores_history)
        # Ensure each subsequent score is no greater than the previous minimum
        cumulative_min_scores = np.minimum.accumulate(scores_array)
        # Plot the objective value over iterations
        plt.figure(figsize=(8, 6))
        plt.plot(cumulative_min_scores, marker='o')
        plt.xlabel('Iterations')
        plt.ylabel('Objective Value')
        plt.title('Convergence Plot')
        plt.grid(True)

        # Save the convergence plot
        plt.savefig(filename)
        plt.close()

    def optimize(self,n_trials, n_init, seed, config_init, order_list):
        """
        Runs the optimization process using the specified search spaces and configurations.
        Set mode to None to cancel test mode.
        """

        def log_time(start, message):
            end = time.time()
            print(f"{message}: {end - start:.4f} seconds")

        err = None
        dup = None
        
        
           
        
        while True:
            try:
                start_time = time.time()
                smpl= self._param_update(self.task_config, self.history, err,dup)
                log_time(start_time, "Parameter update")
                optimizer = self.create_optimizer(smpl)
                get_init_configs(optimizer,self.objective, self.search_spaces, n_init,  seed, config_init)  
                
                break
            except ValueError as e:
                print("OMGerror",e)
                # err = e
                dup=f"## sampler:{smpl} ##"   
        
        while True:
            try:
                dup=None
                start_time = time.time()
                smpl= self._param_update(self.task_config, self.history, err,dup)
                log_time(start_time, "Parameter update")
                optimizer = self.create_optimizer(smpl)
                break
            except ValueError as e:
                print('optimization creation failed:',e)
                err = e

        best_params = None
        best_score = float('inf') if self.direction == "minimize" else -float('inf')

        # Initialize history lists
        params_history = []
        scores_history = []
        i = 0
        while True:
            if i >= n_trials:
                break
            
            print(f"Trial {i + 1}/{n_trials}")
            err = None
            dup = None
            start_time = time.time()
            smpl = self._param_update(self.task_config, self.history, e=err,dup=dup)
            log_time(start_time, "Parameter update (loop)")

            # TODO for optuna
            #optimizer.acq_func = af
            #optimizer.specs['args']['acq_func'] = af
            optimizer.sampler=self.sampler_map[smpl]
            try:
                _run_trial(optimizer,self.objective,catch=())
                # print(optimizer.best_trial.values[0])
                score=optimizer.best_trial.values[0]
                params_dict = optimizer.best_trial.params

                if self.direction == "maximize":
                    score = -score

                self.history['records'].append({"smpl": smpl, "performance": score})
                start_time = time.time()
                err = None
                dup = None
                i += 1

            except Exception as e:
                err = e
                dup = f"## Sampler: {smpl}; ##"
                print('optimization failed with:',e)
                continue

            log_time(start_time, "Optimizer tell")

            # Record history
            params_history.append(params_dict)
            scores_history.append(score)

            if (self.direction == "minimize" and score < best_score) or \
                    (self.direction == "maximize" and score > best_score):
                best_score = score
                self.history['best_score'] = score
                best_params = params_dict

            # save history
            self.params_history = params_history
            self.scores_history = scores_history
        # Plot and save results
        # self.plot_optimization_results(params_history,scores_history, filename=os.path.join('./','results','imgs',f'llmopt.png'))

        return best_params, (-best_score if self.direction == "maximize" else best_score), self.history

def get_init_configs(study,fun_to_evaluate, config_space, n_init,  seed=0, config_init=None):

    if config_init is None or not isinstance(config_init, list):
        raise ValueError("config_init must be a list of valid initial configurations")
    

    data = bo_tpe(fun_to_evaluate, config_space, 0, n_init, seed, config_init, just_fetch_information=True, reset_info=False)
    
    distributions, names = dict(), []
    for hp in config_space.get_hyperparameters():
        name = hp.name
        names.append(name)
        if isinstance(hp, CS.CategoricalHyperparameter):
            distributions[name] = optuna.distributions.CategoricalDistribution(choices=hp.choices)
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            distributions[name] = optuna.distributions.FloatDistribution(low=hp.lower, high=hp.upper, log=hp.log)
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            distributions[name] = optuna.distributions.IntDistribution(low=hp.lower, high=hp.upper)
        else:
            raise TypeError(f"{type(type(hp))} is not supported")

    study.add_trials([
        optuna.create_trial(
            params={name: data[name][i] for name in names},
            distributions = distributions,
            value = data["loss"][i],
        )
        for i in range(n_init)
    ])



    
def optunaexpllmopt(fun_to_evaluate, config_space, n_runs, n_init, seed, task_name, model_name, metric_name,config_init = None, order_list = None):
    if hasattr(fun_to_evaluate, "reseed"):
        fun_to_evaluate.reseed(seed)
    fun_to_evaluate.reset_results()
    np.int = np.int32
    np.float = np.float64
    
    if isinstance(config_space, list):
        raise ValueError("Expected config_space to be a ConfigSpace.ConfigurationSpace, but got a list.")


    
    llm_optimizer = OptunaExpLLMOptimizer(
        objective_function=fun_to_evaluate,  
        # objective_function=obj,
        search_spaces=config_space,          
        task_config={"task": task_name, "model": model_name, "metric": metric_name}, 
        model_path="/root/autodl-tmp/vicuna", 
        direction="minimize"                   
    )

    best_params, best_score, scores_history = llm_optimizer.optimize(n_trials=n_runs, n_init=n_init, seed=seed, config_init=config_init, order_list=order_list)


    final_y = best_score  # 
    print("best_score:",best_score)
    print("history:",scores_history)
    print("llm_result:",final_y)
    return final_y, pd.DataFrame(fun_to_evaluate.all_results)