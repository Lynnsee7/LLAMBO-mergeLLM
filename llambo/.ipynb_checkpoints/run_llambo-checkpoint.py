from llambo.llambo import LLAMBO




task_context = {
    'model': 'RandomForest', 
    'task': 'regression', 
    'tot_feats': 30, 
    'cat_feats': 0, 
    'num_feats': 30, 
    'n_classes': 2, 
    'metric': 'accuracy', 
    'lower_is_better': False, 
    'num_samples': 455, 
    'hyperparameter_constraints': {
        'max_depth': ['int', 'linear', [1, 15]],        # [type, transform, [min_value, max_value]]
        'max_features': ['float', 'logit', [0.01, 0.99]], 
        'min_impurity_decrease': ['float', 'linear', [0.0, 0.5]], 
        'min_samples_leaf': ['float', 'logit', [0.01, 0.49]], 
        'min_samples_split': ['float', 'logit', [0.01, 0.99]], 
        'min_weight_fraction_leaf': ['float', 'logit', [0.01, 0.49]]
    }
}

class BayesmarkExpRunner:
    def __init__(self, task_context, dataset, seed):
        self.seed = seed
        self.model = task_context['model']
        self.task = task_context['task']
        self.metric = task_context['metric']
        self.dataset = dataset
        self.hyperparameter_constraints = task_context['hyperparameter_constraints']
        self.bbox_func = get_bayesmark_func(self.model, self.task, dataset['test_y'])
    
    def generate_initialization(self, n_samples):
        '''
        Generate initialization points for BO search
        Args: n_samples (int)
        Returns: init_configs (list of dictionaries, each dictionary is a point to be evaluated)
        '''

        # Read from fixed initialization points (all baselines see same init points)
        init_configs = pd.read_json(f'bayesmark/configs/{self.model}/{self.seed}.json').head(n_samples)
        init_configs = init_configs.to_dict(orient='records')

        assert len(init_configs) == n_samples

        return init_configs
        
    def evaluate_point(self, candidate_config):
        '''
        Evaluate a single point on bbox
        Args: candidate_config (dict), dictionary containing point to be evaluated
        Returns: (dict, dict), first dictionary is candidate_config (the evaluated point), second dictionary is fvals (the evaluation results)
        '''
        np.random.seed(self.seed)
        random.seed(self.seed)

        X_train, X_test, y_train, y_test = self.dataset['train_x'], self.dataset['test_x'], self.dataset['train_y'], self.dataset['test_y']

        for hyperparam, value in candidate_config.items():
            if self.hyperparameter_constraints[hyperparam][0] == 'int':
                candidate_config[hyperparam] = int(value)

        if self.task == 'regression':
            mean_ = np.mean(y_train)
            std_ = np.std(y_train)
            y_train = (y_train - mean_) / std_
            y_test = (y_test - mean_) / std_

        model = self.bbox_func(**candidate_config)
        scorer = get_scorer(self.metric)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            S = cross_val_score(model, X_train, y_train, scoring=scorer, cv=5)
        cv_score = np.mean(S)
        
        model = self.bbox_func(**candidate_config)  
        model.fit(X_train, y_train)
        generalization_score = scorer(model, X_test, y_test)

        if self.metric == 'neg_mean_squared_error':
            cv_score = -cv_score
            generalization_score = -generalization_score

        return candidate_config, {'score': cv_score, 'generalization_score': generalization_score}


if __name__ == '__main__':

    dataset = 'diabetes'
    seed = 0
    chat_engine = 'gpt-turbo-3.5'
    # LLM Chat Engine, currently our code only supports OpenAI LLM API

    # load data
    pickle_fpath = f'bayesmark/data/{dataset}.pickle'
    with open(pickle_fpath, 'rb') as f:
        data = pickle.load(f)

    # instantiate BayesmarkExpRunner
    benchmark = BayesmarkExpRunner(task_context, data, seed)

    # instantiate LLAMBO
    llambo = LLAMBO(task_context, sm_mode='discriminative', n_candidates=10, n_templates=2, n_gens=10, 
                    alpha=0.1, n_initial_samples=5, n_trials=25, 
                    init_f=benchmark.generate_initialization,
                    bbox_eval_f=benchmark.evaluate_point, 
                    chat_engine=chat_engine)
    llambo.seed = seed

    # run optimization
    configs, fvals = llambo.optimize()