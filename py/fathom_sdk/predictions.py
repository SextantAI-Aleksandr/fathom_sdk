# predictions.py 
# This module is concerned with representing predictions for stock price movement based on various models
# This module does not MAKE the actual predictions themselves, but gives ergonimic classes to represent those predictions.
# The actual heavy-lifting of making predictions is executed on SextantAI servers, by running the architectures
# described in architectures.py with various combinations hyperparameters and input and output stocks. 


from typing import List



class FuturePrice: 
    # The FuturePrice class represents a prediction from ONE model, from ONE date, for ONE price at days_ahead in the future
    def __init__(self, days_ahead: int, future_date: str, sigma: float, pct_chg: float, future_price: float):
        self.days_ahead: int = days_ahead   # The number of trading days in the future for which the prediction was made 
        self.future_date: str = future_date # The date days_ahead in the future, formatted as '2023-12-31' etc. 
        self.sigma: float = sigma           # The model output tensor gives sigma levels for price change 
        self.pct_chg: float = pct_chg       # The percentage change in stock price, given by sigma*std_dev 
        self.future_price: float = future_price # The predicted future price, given by current_price*(100+pct_chg)/100

    def __repr__(self):
        return 'FuturePrice({:.1f}% in {}days)'.format(self.pct_chg, self.days_ahead)

class DeltaSet:
    # The DeltaSet reflects the fact that many models are trained to predict the price movement for a stock at more than one
    # days_ahead time duration in the future. This helps to ensure you are training to capture a signal about price action,
    # and not just overfitting to an artifact on one timescale 
    def __init__(self, from_date: str, closing_price: float, deltas: List[FuturePrice]):
        self.from_date: str = from_date             # The model predictions based on the sequence of closing prices up to this date (inclusive)
        self.closing_price: float = float           # The closing price for the specified stock on from_date
        self.deltas: List[FuturePrice] = deltas     # Predictions at one or more days_ahead intervals in the future 
    
    def __repr__(self):
        l = ', '.join(['{}'.format(fp) for fp in self.deltas ])
        return 'DeltaSet({} -> {})'.format(self.from_date, l)


class PredictionHistory:
    # The prediction class captures multiple DeltaSets of predicted price movements from one model, with one entry
    # for each date on which the closing prices were used to make a prediction
    def __init__(self, pmut_id: str, architecture: str, input_tickers: List[str], output_ticker: str, pred_history:List[DeltaSet]):
        self.pmut_id: str = pmut_id 
        self.architecture: str = architecture            # A string uniquely identifying the architecture used in this model 
        self.input_tickers: List[str] = input_tickers    # the list of input tickers used for this model 
        self.output_ticker: str = output_ticker          # the ticker for the stock being prediced
        self.pred_history: List[DeltaSet] = pred_history # A list of prediction DeltaSets by closing date

    def __repr__(self):
        return 'PredictionHistory({}: {}->{})'.format(self.pmut_id, self.input_tickers, self.output_ticker)

