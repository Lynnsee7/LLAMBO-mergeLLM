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
from skopt import Optimizer
from sampler import obtain_space
from skopt.space import Real, Integer, Categorical
from bo_models.bo_tpe import bo_tpe
from peft import PeftModel


class bops_model_exp:
    def __init__(self, model_path, temperature=0.6):

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model=AutoModelForCausalLM.from_pretrained('/root/LLMBOPS-main/model/vicuna-7b-v1.5',device_map='auto')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature

        
    def generate_response(self, prompt, max_new_tokens=128):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, temperature=self.temperature, do_sample=True)
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response

    
    def _call_model(self, task_config, history, e, dup=None):
        print("now calling model=============================")
        
        self.data = "You are a powerful Bayesian optimizer capable of wisely deciding the best combination of optimization parameters based on the historical information from Bayesian optimization."
        
        allowed_params = 'surrogate_model:GP, RF, ET, GBRT\nacquisition_function:EI,PI,LCB\nacquisition_optimizer:sampling,lbfgs'
        
        target_score = 'a fairly good performance'
        
        if history['records'] == []:
            # target score
            target_score='a fairly good performance'
            history_str = 'Performance:fairly good.\n Hyperparameter configuration:'
        else:
            if history['best_score'] > 0:
                target_score = history['best_score'] * 0.9
            elif history['best_score'] < 0:
                target_score = history['best_score'] * 1.1

            history_configs = []
            cus_history={}
            cus_history = copy.deepcopy(history)
            if len(cus_history['records']) > 5:
                cus_history['records'] = cus_history['records'][-5:-1]
                
            for i in range(len(cus_history['records'])):
                record=cus_history['records'][i]
                entry=f"--Step {i}-- Performance: {record['performance']}\nHyperparameter configuration: ## surrogate model:{record['sm']};  acquisition function:{record['af']};  acquisition optimizer:{record['ao']}; ##"
                history_configs.append(entry)
            history_configs.append(f"--Step {len(cus_history['records'])}-- Performance: {target_score}\nHyperparameter configuration:")
            history_str='\n'.join(history_configs)


        prompt_user = f"""
        \nThe following are examples of the performance of a {task_config['model']} measured in {task_config['metric']} and the corresponding Bayesian optimizer hyperparameter configurations. The model is evaluated on {task_config['task']}. The allowable ranges for the optimizer hyperparameters are: {allowed_params}. Recommend a configuration (one surrogate model, one acquisition function, and one acquisition optimizer) that can achieve the target performance of {target_score}. 
        """
        if e is not None:
            error_information=f"when selecting params,please avoid this error:{str(e)}\n"
            print("error_information",error_information)
            prompt_user+=error_information

            
        prompt_user += "Your response is in one line in the format '## surrogate model: ...; acquisition function: ...; acquisition optimizer: ...; ##'.\n"
        self.data += prompt_user
        
        if dup is not None:
            self.data += f"When selecting params, be different from this combination: {dup}\n"
        self.data += history_str
        

        original_response = self.generate_response(self.data)
        responses = [item.strip() for item in original_response.strip().split('\n') if len(item.strip()) > 0][::-1]
        print("resp:",responses)
        response = None

        for item in responses:
            if item[0]=='#':
                response=item
                break
        if response==None:
            response=' '.join(responses)
        try:
            return response
        except Exception as e:
            print('content generation failed with ',e)
            

    def _parse_params(self, resp):
        match = re.search(r"surrogate[\\\s]?[_\s]?model:\s*([^;]+);\s*acquisition[\\\s]?[_\s]?function:\s*([^;]+);\s*acquisition[\\\s]?[_\s]?optimizer:\s*([a-zA-Z\s]+)", resp)
        
        if match:
            sm = match.group(1).strip()
            af = match.group(2).strip()
            ao = match.group(3).strip()
            return sm, af, ao
        
        else:
            print('Invalid response format:', '----',resp,'----')
            raise ValueError('Failed to parse parameters from response.')

    def __call__(self, task_config, history, e, dup):
        resp = self._call_model(task_config, history, e, dup)
        sm, af, ao = self._parse_params(resp)
        return sm, af, ao



class ExpLLMOptimizer:
    def __init__(self, objective_function, search_spaces, task_config, model_path="/root/LLMBOPS-main/model/vicuna-7b-v1.5", direction="minimize"):
        self.objective = objective_function
        self.search_spaces = search_spaces
        print('----------load model------------')
        self.model = bops_model_exp(model_path)
        self.direction = direction
        self.history = {
            'records': [],
            'best_score': 0
        }
        self.task_config = task_config


        self.scores_history = None
        self.params_history = None

    def _param_update(self, task_config, history, e=None, dup=None):
        while True:
            try:
                sm, af, ao = self.model(task_config, history, e, dup=dup)
                sm = self._validate_base_estimator(sm)
                af = self._validate_acq_func(af)
                ao = self._validate_acq_optimizer(ao)
                break
            except Exception as err:
                print(f"Error during parameter update: {err}")
                e = err
        print(f'{sm},{af},{ao}')
        return sm, af, ao


    
    def _validate_base_estimator(self, base_estimator):
        valid_estimators = ['GP', 'RF', 'ET', 'GBRT']
        if base_estimator not in valid_estimators:
            raise ValueError(f"Invalid base_estimator '{base_estimator}'. Must be one of {valid_estimators}.")
        return base_estimator

    
    def _validate_acq_func(self, acq_func):
        valid_acq_funcs = ['LCB', 'EI', 'PI']
        if acq_func not in valid_acq_funcs:
            raise ValueError(f"Invalid acq_func '{acq_func}'. Must be one of {valid_acq_funcs}.")
        return acq_func

    def _validate_acq_optimizer(self, acq_optimizer):
        valid_acq_optimizers = ['sampling', 'lbfgs', 'auto']
        if acq_optimizer not in valid_acq_optimizers:
            raise ValueError(f"Invalid acq_optimizer '{acq_optimizer}'. Must be one of {valid_acq_optimizers}.")
        return acq_optimizer

    
    
    
    def create_optimizer(self, params, order_list):
        
     
        space   = obtain_space(self.search_spaces, order_list)
        
        optimizer = Optimizer(
            dimensions=space,
            base_estimator=params[0],
            acq_func=params[1],
            acq_optimizer=params[2]
        )
        return optimizer

    

            
    
    def optimize(self, n_trials=100, n_init=5, seed=0, config_init=None, order_list=None):
        def log_time(start, message):
            end = time.time()
            print(f"{message}: {end - start:.4f} seconds")

        err = None
        dup = None

        x0, y0 = get_init_configs(self.objective, self.search_spaces, n_init, order_list, seed, config_init)

        while True:
            try:
                start_time = time.time()
                print(dup)
                sm, af, ao = self._param_update(self.task_config, self.history, err, dup=dup)
                err = None
                dup = None
                log_time(start_time, "Parameter update")

                params = (sm, af, ao)
                optimizer = self.create_optimizer(params, order_list)

                print("params:", params)

                # 初始点插入优化器
                for i in range(len(x0)):
                    print("11111x0[i]", x0[i], "11111y0[i]", y0[i])
                    optimizer.tell(x0[i], y0[i])
                    print("22222x0[i]", x0[i], "22222y0[i]", y0[i])

                break
            except ValueError as e:
                print("OMGerror", e)
                dup = f"## surrogate model: {sm}; acquisition function: {af}; acquisition optimizer: {ao}; ##"

        best_params = None
        best_score = float('inf') if self.direction == "minimize" else -float('inf')
        params_history, scores_history = [], []

        i = len(x0)

        err, dup = None, None
        n_trials += n_init

        while i < n_trials:
            print(f"Trial {i + 1}/{n_trials}")
            start_time = time.time()

            sm, af, ao = self._param_update(self.task_config, self.history, err, dup=dup)
            log_time(start_time, "Parameter update (loop)")

            optimizer.acq_func = af
            optimizer.cand_acq_funcs_ = [af]
            optimizer.specs['args']['acq_func'] = af

            optimizer.acq_optimizer = ao
            optimizer.specs['args']['acq_optimizer'] = ao
            optimizer.specs['args']['base_estimator'] = sm

            try:
                start_time = time.time()
                params = optimizer.ask()

                # 确保顺序一致的映射，基于名称进行映射而非顺序
                param_names = list(self.search_spaces.keys())
                param_dict = {name: param for name, param in zip(param_names, params)}

                # 修正生成的参数，确保其在合法范围内
                for param_name in param_names:
                    search_space_param = self.search_spaces[param_name]
                    param = param_dict[param_name]

                    lower = search_space_param.lower
                    upper = search_space_param.upper

                    # 如果参数值不在合法范围内，进行修正
                    if param < lower or param > upper:
                        print(f"参数 {param_name} 不在范围内 ({param}，范围: [{lower}, {upper}])，重新调整...")
                        if isinstance(search_space_param, CS.hyperparameters.UniformFloatHyperparameter):
                            param_dict[param_name] = np.random.uniform(lower, upper)
                        elif isinstance(search_space_param, CS.hyperparameters.UniformIntegerHyperparameter):
                            param_dict[param_name] = np.random.randint(lower, upper + 1)


                params = [param_dict[name] for name in param_names]

                log_time(start_time, "Optimizer ask")

                # 传递修正后的参数给目标函数
                score = self.objective(param_dict)
                log_time(start_time, "Objective function evaluation")

                if self.direction == "maximize":
                    score = -score

                self.history['records'].append({
                    "sm": sm,
                    "af": af,
                    "ao": ao,
                    "params": param_dict,
                    "performance": score
                })

                optimizer.tell(params, score)

                err = None
                dup = None
                i += 1
            except Exception as e:
                err = e
                print(e)
                dup = f"## surrogate model: {sm}; acquisition function: {af}; acquisition optimizer: {ao}; ##"
                continue

            log_time(start_time, "Optimizer tell")

            params_history.append(params)
            scores_history.append(score)

            if (self.direction == "minimize" and score < best_score) or \
               (self.direction == "maximize" and score > best_score):
                best_score = score
                self.history['best_score'] = score
                best_params = param_dict

 
            self.params_history = params_history
            self.scores_history = scores_history

        return best_params, (-best_score if self.direction == "maximize" else best_score), scores_history
    
    




def get_init_configs(fun_to_evaluate, config_space, n_init, order_list, seed=0, config_init=None):

    if config_init is None or not isinstance(config_init, list):
        raise ValueError("config_init must be a list of valid initial configurations")
    

    init_data = bo_tpe(fun_to_evaluate, config_space, 0, n_init, seed, config_init, just_fetch_information=True, reset_info=False)
    

    y0 = list(init_data["loss"])
    x0 = []
    print("x0:",x0,"y0:",y0)



    for idx in range(n_init):
        this_x = config_init[idx] 
        new_array = []
        for key in order_list:
            param_value = this_x[key]
            lower = config_space[key].lower
            upper = config_space[key].upper

            print(f"Checking param: {key}, Value: {param_value}, Bounds: [{lower}, {upper}]")


            if not (lower <= param_value <= upper):
                print(f"Parameter {key} = {param_value} is out of bounds [{lower}, {upper}], regenerating...")
                
                raise ValueError(f"Generated point for {key} is out of bounds: {param_value} not in [{lower}, {upper}]")

            new_array.append(param_value)
        x0.append(new_array)

    return x0, y0



    
def expllmopt(fun_to_evaluate, config_space, n_runs, n_init, seed, task_name, model_name, metric_name,config_init = None, order_list = None):
    if hasattr(fun_to_evaluate, "reseed"):
        fun_to_evaluate.reseed(seed)
    fun_to_evaluate.reset_results()
    np.int = np.int32
    np.float = np.float64
    
    if isinstance(config_space, list):
        raise ValueError("Expected config_space to be a ConfigSpace.ConfigurationSpace, but got a list.")


    
    llm_optimizer = ExpLLMOptimizer(
        objective_function=fun_to_evaluate,  
        # objective_function=obj,
        search_spaces=config_space,          
        task_config={"task": task_name, "model": model_name, "metric": metric_name}, 
        model_path="/root/LLMBOPS-main/model/vicuna-7b-v1.5", 
        direction="minimize"                   
    )
    print("task:", task_name, "model:", model_name, "metric:", metric_name)

    best_params, best_score, scores_history = llm_optimizer.optimize(n_trials=n_runs, n_init=n_init, seed=seed, config_init=config_init, order_list=order_list)


    final_y = best_score  # 
    print("best_score:",best_score)
    print("history:",scores_history)
    print("llm_result:",final_y)
    return final_y, pd.DataFrame(fun_to_evaluate.all_results)
