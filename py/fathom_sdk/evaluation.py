''' evaluation.py
This library contains things to help with model evaluation.
'''

from typing import List, Optional
from time import time 
from hashlib import sha1
from io import BytesIO
import numpy as np
import statistics
import torch



#~~~Model~Fit~Evaluation~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def r2(tensor_1, tensor_2) -> float:
    # why you want coefficent of determination:
    #    https://towardsdatascience.com/r%C2%B2-or-r%C2%B2-when-to-use-what-4968eee68ed3
    #    ! But I am not setting the intercept to 0!!!!!!
    # how to calculate Coefficient of Determination:
    #    https://www.askpython.com/python/coefficient-of-determination
    corr_matrix = np.corrcoef(
        torch.flatten(tensor_1).detach().numpy(),
        torch.flatten(tensor_2).detach().numpy())
    corr = corr_matrix[0,1]
    codet = corr**2
    return codet

def mabs(tensor):
    # return the median absolute value of a tensor
    return tensor.square().sqrt().median()

def percent_bs(y_actual, y_predicted) -> float:
    # given an actual output and a preicted output,
    # return the 'percent bullsh**' of the numbers you are predicting
    bs = float(100*mabs(y_actual - y_predicted )/(0.001 + mabs(y_actual)))
    return bs

def recent_stdev(arr, n=20):
    # return the 'recent standard deviation)
    try:
        stdev = statistics.stdev(arr[-n:])
    except statistics.StatisticsError:
        stdev = 1e-7 # this is return if there is only one data point
    return stdev
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~TRAINING~EXIT~CODES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NEW_BEST_FIT =          'new_best_fit'  # not actually an exit code- this means save because this is the new best
EXIT_OVERFIT_R2 =       'overfit_r2'    # OVERFITTING on r2 >> r2_cv
EXIT_UNDERFIT_R2 =      'underfit_r2'   # UNDERFITTING on r2_cv
EXIT_STAGNANT_LOSS =    'stagnant_ls'   # STAGNANT LOSS: pct_loss_stdev ~= 0
EXIT_STAGNANT_R2 =      'stagnant_r2'   # STAGNANT R2: core.recent_stdev(history_r2)/r2 ~= 0
EXIT_NUMPY_NAN =        'numpy_nan'     # NUMPY NAN encountered
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



class BestModel:
    # This class contains the things you need to save a model binary to memory
    def __init__(self, model):
        buff = BytesIO()
        torch.save(model.state_dict(), buff)
        best_model = buff.getvalue()
        self.sha1 = sha1(best_model).hexdigest()
        self.state_dict_bytes: bytes = best_model


class Metrics:
    # The Metrics class takes a set of actual and predicted y values 
    # and captures various metric that indicate how good the fit is
    def __init__(self, criterion, y_pred, y_act):
        loss_object = criterion(y_pred, y_act)
        self.loss: float = loss_object.item()
        self.r2: float = r2(y_pred, y_act)
        self.pct_bs: float = percent_bs(y_act, y_pred)


class GradientDescentState:
    # The state of gradient descent is captured by the epoch 
    def __init__(self, epoch: int, criterion, y_pred, y_act, y_cv_pred, y_cv_act, elapsed: float):
        self.epoch: int = epoch
        self.elapsed: float = elapsed   # seconds 
        self.train = Metrics(criterion, y_pred, y_act)
        self.cv = Metrics(criterion, y_cv_pred, y_cv_act)
            

class GradientDescentHistory:
    def __init__(self):
         self.time_init: float = time()
         self.elapsed: float = 0.0
         self.history: List[GradientDescentState] = []
         self.best_epoch: int = 0    # epoch with the highest r2_cv
         self.best_pct_bs_cv: float = 999999
         self.best_model: Optional[BestModel] = None 
    
    def update(self, model, epoch, criterion, y_pred, y_act, y_cv_pred, y_cv_act, summary:str, report=True) -> Optional[str]:
        # update metrics for the main and CV lists, returning any exit code
        self.elapsed = time() - self.time_init
        gds = GradientDescentState(epoch, criterion, y_pred, y_act, y_cv_pred, y_cv_act, self.elapsed)
        if (str(gds.train.r2) == 'nan' or str(gds.cv.r2) == 'nan'):
            # If a numpy NaN is encountered, exit immediately so you don't append gds to history and fill Postgres with NaN
            return  EXIT_NUMPY_NAN
        self.history.append(gds)
        # calculate metrics relating to the best loss
        self.min_loss: float = min([ x.train.loss for x in self.history])
        self.min_cv_loss: float = min([ x.cv.loss for x in self.history])
        # calculate metrics related to how much loss and r2 are changing- are they becoming stagnant?
        self.pct_loss_stdev = 100*recent_stdev([ x.train.loss for x in self.history])/(gds.train.loss + 1e-7) # what percent of loss is its standard deviation
        self.pct_r2_stdev = 100*recent_stdev([ x.train.r2 for x in self.history])/(gds.train.r2 + 1e-7) # what percent of r2 is its standard deviation
        # print a line representing progress
        if report:
            print('  epoch={}   elapsed={:.0f}s   r2/cv={:.3f}/{:.3f}   loss/cv={:.3f}/{:.3f}   pct_bs/cv={:.0f}/{:.0f}  {} '.format(epoch, self.elapsed, gds.train.r2, gds.cv.r2, gds.train.loss, gds.cv.loss, gds.train.pct_bs, gds.cv.pct_bs, summary), end='\r')
        # test for various exit conditions
        minimum_try = ( (self.elapsed > 30) and (epoch > 12) )
        if minimum_try and (gds.train.r2 > 0.25) and (gds.cv.r2 < 0.02):
            return EXIT_OVERFIT_R2
        if minimum_try and  (gds.train.r2 > 0.4) and (gds.cv.r2 < 0.05):
            return EXIT_OVERFIT_R2
        if minimum_try and (epoch > 50) and (gds.cv.r2 < 0.02):
            return EXIT_UNDERFIT_R2
        if minimum_try and (epoch > 150) and (gds.cv.r2 < 0.10):
            return EXIT_UNDERFIT_R2
        if minimum_try and (self.elapsed > 60) and (gds.train.r2 < 0.10) and (self.pct_loss_stdev < 0.01):
            return EXIT_STAGNANT_LOSS # its been over a minute, the fit is poor, and loss seems stagnant
        if minimum_try and (self.elapsed > 60) and (gds.train.r2 < 0.10) and (self.pct_r2_stdev < 0.1):
            return EXIT_STAGNANT_R2 # its been over a minute, the fit is poor, and r2 seems stagnant
        # check to see if this is the new best epoch, but only if there was no other exit code
        if (gds.cv.pct_bs < self.best_pct_bs_cv) and (epoch %3 == 0):
            # this means you will return the NEW_BEST_FIT code, but don't do it EVERY single time, hence the epoch %3
            self.best_pct_bs_cv = gds.cv.pct_bs
            self.best_epoch = epoch
            # If the fit is good enough, save the model state so you can persist to disk 
            if gds.cv.pct_bs < 85:
                self.best_model = BestModel(model)
            return NEW_BEST_FIT
        


