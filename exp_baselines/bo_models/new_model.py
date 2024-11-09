import re
import time
import copy
import torch
import random
import numpy as np
import pandas as pd
import os
import json
import string
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import get_conv_template
import optuna
from optuna.samplers import *
from optuna.study._optimize import _run_trial
from .descriptions import descriptions,form_description
from .rewrite import rewrite_function
from fastchat.conversation import get_conv_template
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

class bops_model_optuna:
    def __init__(self,model_path,prompt_type,temperature=0.3):
        # create agent
        self.model_path=model_path
        self.prompt_type=prompt_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,device_map='auto')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # specifically for vicuna model.
        self.conv=get_conv_template('vicuna_v1.1')
        # self.model.to(self.device)
        self.temperature=temperature
        # naive version
        # self.init_prompt_template = """You are a powerful analyzer for analyzing machine learning models.
        # Background:<task_type> task on <dataset_name> dataset.
        # Dataset description:<dataset_desc>
        # Model:<model_name>
        # Below are a set of hyperparameters of this model.Choose one which is most likely to have the best performance.
        # <hyperparams>
        # (keep your response as concise as possible and strictly follow the instructions)Think step by step.First,analyze the task and infer the demand of expected params.Then,look through the parameters based on model ,dataset and task and tell whether they are suitable.Then,select the parameters.Finally,output the **index** of your selected hyperparameters such as 'A'/'B'/'C'..."""
        # self.followup_prompt_template="""Under this hyperparameter settings,the model's performance is <performance>.Please choose one new set of hyperparameter from below which is most possible of having better performance.
        # <hyperparams>
        # (keep your response as concise as possible and strictly follow the instructions)Think step by step.First,analyze the task and infer the demand of expected params.Then,look through the parameters based on model ,dataset and number of iteration and tell whether they are suitable.Then,select the parameters.Finally,output the **index** of your selected hyperparameterssuch as 'A'/'B'/'C')"""
        
        # finetuned version
        self.init_prompt_template=self.followup_prompt_template="""\n            Task description: This is the configuration of a machine learning task.<dataset_desc>\n\n            Model name: <model_name>\n            Model description:<model_desc>\n            Here is a set of hyperparameters for the machine learning model on this task, each has a different performance:<hyperparams>Which of the following hyperparameter combinations is the best choice? Please analyze the task, dataset and model, and give a detailed explanation first, then output your final decision in one single line.\n"""

    def generate_response(self,prompt, max_new_tokens=2048):
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    #     outputs = self.model.generate(inputs.input_ids, max_new_tokens=max_new_tokens,temperature=self.temperature,do_sample=True)
    #     new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    #     print('token count: ',len(new_tokens))
    #     response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    #     return response
        os.environ["OPENAI_API_KEY"] = "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"
        os.environ["OPENAI_API_VERSION"] = "2020-10-01"
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
        os.environ["OPENAI_API_TYPE"] = "open_ai"
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():  # 避免保存计算图
                outputs = self.model.generate(
                    inputs.input_ids, 
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True
                )

            # 立即将输出转移到CPU并释放GPU变量
            new_tokens = outputs[0][inputs.input_ids.shape[1]:].cpu()
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # 显式删除中间变量
            del inputs
            del outputs
            del new_tokens
            return response
        
        finally:
            # 确保清理中间变量
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    # def _parse_hyperparams(self,hyperparams_map):
    #     hyperparams_str = ""
    #     alphabet = string.ascii_uppercase
    #     for i, hyperparam in enumerate(hyperparams_map.keys()):
    #         hyperparams_str += f"{alphabet[i]}. {hyperparam}\n"
    #     return hyperparams_str

    def _parse_hyperparams(self, hyperparams_map):
        hyperparams_str = ""
        alphabet = string.ascii_uppercase

        print("Debug - hyperparams_map keys:", list(hyperparams_map.keys()))

        for i, hyperparam_str in enumerate(hyperparams_map.keys()):
            try:
                params = json.loads(hyperparam_str)  # 解析存储的JSON字符串
                params_str = "\n".join([f"  {k}: {v}" for k, v in params.items()])
                hyperparams_str += f"{alphabet[i]}.\n{params_str}\n\n"
                # print(f"Formatted option {alphabet[i]}:", params_str)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {hyperparam_str}")
                continue

        return hyperparams_str
    
    
    def _call_model(self,task_config,hyperparams_map,last_score,e=None,dup=None):
        hyperparams=self._parse_hyperparams(hyperparams_map)
        print("hyperparams:",hyperparams)
        if self.conv.messages==[]:#init
            # print(task_config)
            init_prompt=copy.copy(self.init_prompt_template).replace('<task_type>',task_config['task_type'])\
                                                            .replace('<dataset_name>',task_config['dataset_name'])\
                                                            .replace('<dataset_desc>',task_config['dataset_desc'])\
                                                            .replace('<model_name>',task_config['model_name'])\
                                                            .replace('<model_desc>',task_config['model_desc'])\
                                                            .replace('<hyperparams>',hyperparams)
            self.conv.append_message(self.conv.roles[0],init_prompt)
        else:
            followup_prompt=copy.copy(self.followup_prompt_template).replace('<task_type>',task_config['task_type'])\
                                                                    .replace('<dataset_name>',task_config['dataset_name'])\
                                                                    .replace('<dataset_desc>',task_config['dataset_desc'])\
                                                                    .replace('<model_name>',task_config['model_name'])\
                                                                    .replace('<model_desc>',task_config['model_desc'])\
                                                                    .replace('<hyperparams>',hyperparams)\
                                                                    .replace('<performance>',str(last_score))\
                                                                    .replace('<hyperparams>',hyperparams)
            self.conv.append_message(self.conv.roles[0],followup_prompt)
        self.conv.append_message(self.conv.roles[1],None)

        prompt=self.conv.get_prompt()
        original_response = self.generate_response(prompt)
        return original_response
            
    def find_index(self,resp):
        for c in resp[::-1]:
            if c in string.ascii_uppercase:
                return c
        return None
    def _parse_response(self,resp,hyperparams_map):
        print('resp of llm:  ','\n----------------------------------------------------------------\n',
                  resp,'\n----------------------------------------------------------------\n')
        idx=self.find_index(resp)
        for i, hyperparam in enumerate(hyperparams_map.keys()):
            if idx == string.ascii_uppercase[i]:
                print('selected hyperparam:',hyperparam)
                return hyperparam
        raise ValueError(f"Invalid response '{resp}'. Must be one of {string.ascii_uppercase[:len(hyperparams_map.keys())]}.")
    
    def get_token_count(self, text):
        """
        Returns the number of tokens in the given text.
        
        Parameters:
        - text (str): The text to count the tokens of.
        
        Returns:
        - int: The number of tokens in the text.
        """
        return len(self.tokenizer(text)['input_ids'])

    def _truncate_conversation_if_needed(self, context_length=4096):
        """
        (16k is 16384)
        Checks if the conversation exceeds the specified context length and truncates if necessary.
        
        Continually removes the first round of dialogue until the total token count 
        is below the context length.
        
        Parameters:
        - context_length (int): The maximum allowed number of tokens in the conversation.
        """
        current_token_count = self.get_token_count(self.conv.get_prompt())  # Assuming this method exists

        # Loop to remove the first round of conversation until the token count is under the limit
        while current_token_count > context_length*0.75:
            print(f"Conversation exceeds {context_length} tokens. Truncating...")

            # Ensure there are more than one round of messages before we delete
            if len(self.conv.messages) > 1:
                # Remove the first round of dialogue (assuming it's at index 0)
                del self.conv.messages[1]
                print("First round of conversation deleted.")

                # Recalculate the token count after truncation
                current_token_count = self.get_token_count(self.conv.get_prompt())
            else:
                print("Not enough conversation rounds to delete.")
                break  # Exit the loop if there's nothing more to delete

            print('truncate done! token count:',current_token_count)

    def __call__(self, task_config,hyperparams_map,last_score=None, e=None,dup=None):
        # start_time = time.time()
        # print(2)
        
        # end_time = time.time()
        
        # execution_time = end_time - start_time
        # print(f"Execution time for _call_model: {execution_time:.4f} seconds")
        while True:
            try:
                resp = self._call_model(task_config, hyperparams_map,last_score,e,dup)
                selected_hyperparams = self._parse_response(resp,hyperparams_map)
                break
            except Exception as err:
                e=err
                print('Exception:',e,'\nretrying...')
        self.conv.update_last_message(resp)
        # Check if the conversation exceeds 4096 tokens
        self._truncate_conversation_if_needed()
        
        return selected_hyperparams


class OptunaMTLLMOptimizer:
    def __init__(self,
                 objective_function,
                 task_config,
                 task_config_simple,
                #  model_path='/home/lang.gao/proj/models/vicuna-13b',
                # model_path='/mnt/data/users/Lang_Gao/proj/models/vicuna-7b-v1.5-ftmlp-10000sample',
                model_path='/root/autodl-tmp/LLMBOPS_before/model/HeartyHaven/vicuna-7b-v1.5-ftmlp',
                 prompt_type=None,
                 direction="minimize"
                 ):
        print("initialize OptunaLLMOptimizer")
        self.objective = objective_function
        self.model_path=model_path
        self.prompt_type=prompt_type
        # for test ,comment out the following 2 lines
        print(f'----------load model from {model_path}------------')
        self.model = bops_model_optuna(model_path,self.prompt_type)

        self.direction = direction
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
        self.pruner_map = {
            'MedianPruner': optuna.pruners.MedianPruner(),
            'NopPruner': optuna.pruners.NopPruner(),
            'PatientPruner': optuna.pruners.PatientPruner(wrapped_pruner=optuna.pruners.NopPruner(), patience=10),
            'PercentilePruner': optuna.pruners.PercentilePruner(percentile=50.0),  
            'SuccessiveHalvingPruner': optuna.pruners.SuccessiveHalvingPruner(),
            'HyperbandPruner': optuna.pruners.HyperbandPruner(),
            'ThresholdPruner': optuna.pruners.ThresholdPruner(lower=0.01),
            'WilcoxonPruner': optuna.pruners.WilcoxonPruner()
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


    def create_optimizers(self, base_optimizer=None, num_combinations=10):
        """
        Creates an optimizer list instance for a given search space and configuration.
        Combines sampler_map and pruner_map to create multiple optimizers.
        Randomly selects a specified number of optimizer combinations (default 10).
        """
        all_combinations = []

        # Iterate over all combinations of samplers and pruners
        for sampler_name, sampler in self.sampler_map.items():
            for pruner_name, pruner in self.pruner_map.items():
                all_combinations.append((sampler_name, pruner_name))

        # Randomly select num_combinations from the available combinations
        selected_combinations = random.sample(all_combinations, num_combinations)

        Optimizer_list = []
        for sampler_name, pruner_name in selected_combinations:
            sampler = self.sampler_map[sampler_name]
            pruner = self.pruner_map[pruner_name]

            if base_optimizer is None:
                # Create a new study with the sampler and pruner
                optimizer = optuna.create_study(
                    direction=self.direction,
                    sampler=sampler,
                    pruner=pruner,
                    storage=None
                )
            else:
                # Deepcopy the base optimizer and update the sampler and pruner
                optimizer = copy.deepcopy(base_optimizer)
                optimizer.sampler = sampler
                optimizer.pruner = pruner

            # Append the combination (sampler, pruner) and optimizer to the list
            Optimizer_list.append(((sampler_name, pruner_name), optimizer))

        return Optimizer_list

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
        model_name=self.model_path.split('/')[-1]
        if 'dpo' in self.model_path.lower():
            save_file=os.path.join(save_path,f"{model_name}dpo_{self.task_config_simple['task']}_{self.task_config_simple['model']}.txt")
            # Write the points to 'skopt.txt': ablation1-ensure using the best tuned model.
            if self.prompt_type=='detail':
                save_file=os.path.join(save_path,f"{model_name}dpodetail_{self.task_config_simple['task']}_{self.task_config_simple['model']}.txt")
            elif self.prompt_type=='no_history':
                save_file=os.path.join(save_path,f"{model_name}dponohistory_{self.task_config_simple['task']}_{self.task_config_simple['model']}.txt")
                
        elif 'sft' in self.model_path.lower():
            save_file=os.path.join(save_path,f"{model_name}_sft_{self.task_config_simple['task']}_{self.task_config_simple['model']}.txt")
        elif 'vicuna' in self.model_path.lower():
            save_file=os.path.join(save_path,f"{model_name}_{self.task_config_simple['task']}_{self.task_config_simple['model']}.txt")
        with open(save_file, 'w') as f:
            for x, y in points:
                f.write(f"{x},{y}\n")

    def _get_one_hyperparams(self,name,optimizer,map_hyperparams_params):
        while True:
            try:
                # _run_trial auto saves history and other info
                _run_trial(optimizer,self.objective,catch=())
                # get config of this trial
                param=name
                last_trial=optimizer.trials[-1]
                # score=optimizer.best_trial.values[0]
                score=last_trial.values[0]
                hyperparams=last_trial.params
                # hyperparams = optimizer.best_trial.params

                if self.direction == "maximize":
                    score = -score
                # save candidate configs
                # only allow string key
                # print('added!')
                map_hyperparams_params[json.dumps(hyperparams)] = (copy.copy(optimizer),param,score) # may be little anti-intuitive
                break

            except Exception as e:
                print('optimization failed with:',e)
                continue



            
    def optimize(self, n_trials=100, mode='test'):
        """
        Runs the optimization process using the specified search spaces and configurations.
        Set mode to None to cancel test mode.
        """
        # selectable params=GPSampler,TPESampler,CmaEsSampler,RandomSampler,NSGAIISampler,NSGAIIISampler,QMCSampler...
        
        # Initialize optimizer
        optimizer_list = self.create_optimizers()

        best_params = None
        best_score = float('inf') if self.direction == "minimize" else -float('inf')

        # Initialize history lists
        params_history = []
        scores_history = []
        last_score = None
        i = 0
        while True:
            if i >= n_trials:
                break
            if mode == "test":
                print(f"Trial {i + 1}/{n_trials}")

            map_hyperparams_params = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for name, optimizer in optimizer_list:
                    # Submit each optimizer to run in a separate thread
                    futures.append(executor.submit(self._get_one_hyperparams, name, optimizer,map_hyperparams_params))
                
                # Wait for all threads to complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        # If there are any exceptions in the threads, they will be re-raised here
                        future.result()
                    except Exception as e:
                        print(f"Thread raised an exception: {e}")
               

            # for actual use
            print("map_hyperparams_params:",map_hyperparams_params)
            selected_hyperparams=self.model(task_config=self.task_config,hyperparams_map=map_hyperparams_params,last_score=last_score)
            # for test
            # selected_hyperparams=next(iter(map_hyperparams_params.keys()))

            selected_optimizer,selected_param,selected_score=map_hyperparams_params[selected_hyperparams]
            print(f"selected optimizer:{selected_param},selected_hyperparams:{selected_hyperparams},selected_score:{selected_score}")   
            # Record history
            params_history.append((selected_param, selected_hyperparams))
            scores_history.append(selected_optimizer.best_trial.values[0])
            last_score = selected_score
            if (self.direction == "minimize" and selected_score < best_score) or \
                    (self.direction == "maximize" and selected_score > best_score):
                best_score = selected_score
                self.history['best_score'] = selected_optimizer.best_trial.values[0]
                best_hyperparams = selected_hyperparams

            # save history
            self.params_history = params_history
            self.scores_history = scores_history
            # update optimizer list
            del optimizer_list # release memory

            optimizer_list = self.create_optimizers(selected_optimizer)
            i += 1
        
        return best_hyperparams, (-best_score if self.direction == "maximize" else best_score), self.history
    


# def expllmopt(fun_to_evaluate, config_space, n_runs, n_init, seed, task_name, model_name, metric_name, config_init=None, order_list=None):
#     if hasattr(fun_to_evaluate, "reseed"):
#         fun_to_evaluate.reseed(seed)
#     fun_to_evaluate.reset_results()
    
#     # 获取参数引用
#     params_ref = {}
#     for param_name in config_space.get_hyperparameter_names():
#         hp = config_space.get_hyperparameter(param_name)
#         param_type = float if isinstance(hp, UniformFloatHyperparameter) else int
#         params_ref[param_name] = {
#             "type": param_type,
#             "range": [hp.lower, hp.upper],
#             "name": param_name
#         }

#     class WrappedObjective:
#         def __init__(self, fun_to_evaluate):
#             self.fun_to_evaluate = fun_to_evaluate
            
#         def __call__(self, trial):
#             params = {}
#             try:
#                 # 从trial.params获取参数
#                 params = trial.params
#             except AttributeError:
#                 # 如果trial是字典，直接用
#                 params = trial

#             return self.fun_to_evaluate(params)

        
#     dataset_type = 'classification' # 或者根据实际情况设置
#     dataset_name = task_name
    
#     task_config = {
#         "task_type": f"{dataset_type} task on {dataset_name} dataset.",
#         "dataset_name": dataset_name,
#         "dataset_desc": form_description(descriptions, dtype="datasets", 
#                                      name=dataset_name.lower(), 
#                                      extra_features=["statistical_characteristics", 
#                                                    "feature_meaning", "label"]),
#         "model_name": model_name,
#         "model_desc": form_description(descriptions, dtype="models", 
#                                    name=model_name.lower()),
#         "metric": 'accuracy' if dataset_type == 'classification' else 'output value'
#     }
#     # print("=======task_config:", task_config)
    
#     task_config_simple = {
#         'task': task_name,
#         'model': model_name
#     }

#     # 创建优化器，使用包装后的目标函数
#     optimizer = OptunaMTLLMOptimizer(
#         objective_function=WrappedObjective(fun_to_evaluate),  # 使用包装类
#         task_config=task_config,
#         task_config_simple=task_config_simple,
#         # model_path="/root/LLMBOPS-main/model/vicuna-7b-v1.5",
#         model_path="/root/autodl-tmp/LLMBOPS_before/model/HeartyHaven/vicuna-7b-v1.5-ftmlp",
#         # model_path="/root/autodl-tmp/LLMBOPS_before/model/HeartyHaven/vicuna-7b-v1.5-ftmlp-2000sample-singleround",
#         direction="minimize"
#     )
#     # 打印函数信息而不是尝试获取源码
#     print("Function information:")
#     print(f"- Type: {type(fun_to_evaluate)}")
#     print(f"- Callable: {callable(fun_to_evaluate)}")
#     print(f"- Methods: {[m for m in dir(fun_to_evaluate) if not m.startswith('_')]}")
  

#     try:
#         best_hyperparams, best_score, history = optimizer.optimize(n_runs)
#         return best_score, pd.DataFrame(fun_to_evaluate.all_results)
#     except Exception as e:
#         print(f"Optimization failed: {str(e)}")
#         return None, None


def expllmopt(fun_to_evaluate, config_space, n_runs, n_init, seed, task_name, model_name, metric_name, config_init=None, order_list=None):
    # 已有部分
    if hasattr(fun_to_evaluate, "reseed"):
        fun_to_evaluate.reseed(seed)
    fun_to_evaluate.reset_results()
    
    # 获取参数引用
    params_ref = {}
    for param_name in config_space.get_hyperparameter_names():
        hp = config_space.get_hyperparameter(param_name)
        param_type = float if isinstance(hp, UniformFloatHyperparameter) else int
        params_ref[param_name] = {
            "type": param_type,
            "range": [hp.lower, hp.upper],
            "name": param_name
        }
        
    # 关键是这个 WrappedObjective 类需要正确处理参数
    class WrappedObjective:
        def __init__(self, fun_to_evaluate):
            self.fun_to_evaluate = fun_to_evaluate
            self.api_config = params_ref

        def __call__(self, trial):
            # 确保每个参数都被正确设置
            params = {}
            for param_name, param_config in self.api_config.items():
                if param_config['type'] == float:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['range'][0],
                        param_config['range'][1]
                    )
                elif param_config['type'] == int:
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['range'][0],
                        param_config['range'][1]
                    )

            # 记录和返回结果
            result = self.fun_to_evaluate(params)
            print(f"Evaluated parameters: {params} with result: {result}")
            return result

    dataset_type = 'classification' 
    dataset_name = task_name
    
    task_config = {
        "task_type": f"{dataset_type} task on {dataset_name} dataset.",
        "dataset_name": dataset_name,
        "dataset_desc": form_description(descriptions, dtype="datasets", 
                                     name=dataset_name.lower(), 
                                     extra_features=["statistical_characteristics", 
                                                   "feature_meaning", "label"]),
        "model_name": model_name,
        "model_desc": form_description(descriptions, dtype="models", 
                                   name=model_name.lower()),
        "metric": 'accuracy' if dataset_type == 'classification' else 'output value'
    }
    
    task_config_simple = {
        'task': task_name,
        'model': model_name
    }

    # 创建优化器，使用包装后的目标函数
    optimizer = OptunaMTLLMOptimizer(
        objective_function=WrappedObjective(fun_to_evaluate),  # 使用包装类
        task_config=task_config,
        task_config_simple=task_config_simple,
        model_path="/root/autodl-tmp/LLMBOPS_before/model/HeartyHaven/vicuna-7b-v1.5-ftmlp",
        direction="minimize"
    )

    try:
        best_hyperparams, best_score, history = optimizer.optimize(n_runs)
        return best_score, pd.DataFrame(fun_to_evaluate.all_results)
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        return None, None