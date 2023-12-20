''' xform.py
This library deals with the "transformation" of time-sequences of price data. Consider lists of [(date, price), (date, price)]
tupes for several different securities. If you would like to make predictions using that data, consider these needs:
    1) Calculating the changes from one day to the next
    2) Knowing which days to skip becuase they are not trading days, and how to compensate if one list is misssing a date or two
    3) Allowing deltas of something other than a day (i.e. 2,4,5 trading days etc.)
    4) Normalizing price changes on a sigma-basis (i.e. by standard deviations)
    5) Splitting data into train and cross-validation (CV) sets
    6) Adding noise to avoid overfitting

The code in this module allows all of the above described tranformations 

NOTE: set the envionment variable SHOW_SPLITS=true to show splits into train/CV/buffer
'''


from typing import Optional
import torch 
from fathom_sdk.xform_nodeps import TimeConfigCore 


#~~~Random~Matrices~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def rand_mtx(noise_sigma, *shape):
    # generate a random matrix of *shape with elements between [-noise_sigma, noise_sigma)
    return noise_sigma*(-1+2*torch.rand(*shape))

def add_noise(mtx, noise_sigma):
    # add noise to a given matrix mtx by adding a number between [-sigma, sigma) to each value
    if noise_sigma in (0, None):
        return mtx
    noise = rand_mtx(noise_sigma, *mtx.shape)
    return mtx + noise

def augment(mtx, augment_mult, noise_sigma):
    # augment the matrix mtx by concatinating augment_mult copies of the data, each with sigma noise added
    if augment_mult in (1, None):
        return add_noise(mtx, noise_sigma)
    noisy_copies = []
    for _ in range(augment_mult):
        noisy_copies.append(add_noise(mtx, noise_sigma))
    augmented = torch.cat([*noisy_copies], dim=0)
    return augmented
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TimeConfig(TimeConfigCore):
    # The time config specifies an end_date, a number of years of history prior to that date, and a delta size in terms of trading days
    def __init__(self, 
            end_date: str,          # the last trading day to consider, i.e. '2022-11-31'
            years_history: int,     # the number of years of history of trading data you want to use, i.e. 10 etc
            delta_size: int,        # Calculate price changes between n consecutive business days 
            inc_offsets:Optional[bool]=True
            ):
        TimeConfigCore.__init__(self, end_date, years_history, delta_size, inc_offsets=inc_offsets)


    def nosplit_tensors(self, xy, augment_mult=None, noise_sigma=None):
        # convert xy into tensors with no splitting
        # this function is intended for use with predictions where there is no splitting into test, train sets
        x = augment( torch.tensor([ dp['x_seq_sigmas'] for dp in xy ]), augment_mult, noise_sigma)
        y = augment( torch.tensor([ dp['y_future_sigmas'] for dp in xy ]), augment_mult, noise_sigma)
        return x, y
        

    def split_xy_tensors(self, xy, augment_mult=None, noise_sigma=None, **kwargs):
        # split the xy data into train and CV sets
        # then return tensors for each
        xy_train, xy_cv, cv_start_date, cv_end_date = self.split_xy(xy, **kwargs)
        x_train = augment(torch.tensor([ dp['x_seq_sigmas'] for dp in xy_train ]), augment_mult, noise_sigma)
        y_train = augment(torch.tensor([ dp['y_future_sigmas'] for dp in xy_train ]), augment_mult, noise_sigma)
        x_cv = augment(torch.tensor([ dp['x_seq_sigmas'] for dp in xy_cv ]), augment_mult, noise_sigma)
        y_cv = augment(torch.tensor([ dp['y_future_sigmas'] for dp in xy_cv ]), augment_mult, noise_sigma)
        return x_train, y_train, x_cv, y_cv, cv_start_date, cv_end_date