

from typing import List, Dict

# This constant array can be imported for SQL queries 
MODEL_FIELDS = ['msrc_id', 'pmut_id', 'years_history', 'delta_size', 'seq_length', 'architecture', 'args_json', 'kwargs_json', 'weight_decay', 'learning_rate', 'inc_offsets', 'augment_mult', 'noise_sigma', 'mach_id', 'dp_train', 'dp_cv', 'dp_buffer', 'exit_code', 'r2', 'r2_cv', 'pct_bs', 'pct_bs_cv', 'score', 'best_epoch', 'inputs', 'outputs', 'deltas']

class Model:
    def __init__(self, jz: Dict[str,object]):
        # initialize from a JSON dict
        # the source from which model hyperparamers were generated
        self.msrc_id: str = jz['msrc_id']
        # The 'permutation id' uniquely identifying this model 
        self.pmut_id : str = jz['pmut_id']
        # the number of years history of price action, up to and inclusive of self.end_date 
        self.years_history : float = jz['years_history']
        # 1 means take a new data point every trading day, 2 means every 2 trading days, etc.
        self.delta_size : int = jz['delta_size']
        # a list of tickers used as inputs for this model
        # (or symbols for synthetic 'tickers' associated with an abstraction- i.e. a Centroid or Principal Component)
        self.inputs : List[str] = jz['inputs']
        # a list of output tickers (typically just one) for which the model is trying
        # to predict price movements
        self.outputs : List[str] = jz['outputs']
        # The deltas at which future price prediction is attempted
        # For example, a model with .delta_size = 2 and .deltas = [3, 5]
        # Would try to predict price movement 2*[3, 5] = [6, 10] trading days in the future
        self.deltas : List[int] = jz['deltas']
        # a string uniquely identifying the model architecture (sans hyperparameters) used 
        self.architecture : str = jz['architecture']
        # architecture hyperparameter args:
        self.args_json : List[int] = jz['args_json']
        # architecture hypermarameter kwargs: typically {} or {drop_probs=}
        self.kwargs_json : Dict[str, float] = jz['kwargs_json']
        # weight decay used for the torch.optim.Adam optimizer
        # see https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        self.weight_decay : float = float(jz['weight_decay'])
        # learning rate used for used for the torch.optim.Adam optimizer
        # see https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        self.learning_rate : float = jz['learning_rate']
        # include offsets for delta_size > 1 ? i.e. for delta_size=5,
        # do you consider just Friday to Friday price changes, or the 4 other days as well?
        self.inc_offsets : bool = jz['inc_offsets']
        # augmentation multiplier used for data augmentation
        # see xform.py/augment()
        self.augment_mult : int = int(jz['augment_mult'])
        # noise sigma used to reduce overfitting, often used with data augmentation
        # see xform.py/add_noise() and xform.py/augment()
        self.noise_sigma : float = float(jz['noise_sigma'])
        # sequence length of successive sets of sigma levels for input stocks used as input tensors
        self.seq_length: int = int(jz['seq_length'])
        # number of datapoints used in the training set
        self.dp_train : int = int(jz['dp_train'])
        # number of datapoints used in the Cross-Validation set
        self.dp_cv : int = int(jz['dp_cv'])
        # number of datapoints skipped as a 'buffer' between training and Cross-Validation sets
        # That is, if the sequence length is 30, you don't want the last few datapoints toward the end of a sequence for training
        # to be the same as the first few datapoints EARLIER in the sequence for cross-validation
        self.dp_buffer : int = int(jz['dp_buffer'])
        # The exit code given when Sextant AI servers stopped performing gradient descent on this model,
        # givng a very breif explanation of why the model was stopped
        self.exit_code : str = str(jz['exit_code'])
        # r2 metric achieved on the training set
        # higher is better: see evaluation.py/r2()
        self.r2 : float = float(jz['r2'])
        # r2 metric achieved on the cross-validation set
        # higher is better: see evaluation.py/r2()
        self.r2_cv : float = float(jz['r2_cv'])
        # 'Percent Bulls***' metric achieved on the training set
        # lower is better: see evaluation.py/percent_bs()
        self.pct_bs : float = float(jz['pct_bs'])
        # 'Percent Bulls***' metric achieved on the cross-validation
        # lower is better: see evaluation.py/percent_bs()
        self.pct_bs_cv : float = float(jz['pct_bs_cv'])
        # internal scoring metric, higher is better
        self.score : float = float(jz['score'])
        # The number of epochs at which the best (lowest) .pct_bs_cv was obtained
        # in the case of overfitting, the .pct_bs_cv tends to increase after this epoch as overfitting becomes worse
        self.best_epoch : int = int(jz['best_epoch'])

