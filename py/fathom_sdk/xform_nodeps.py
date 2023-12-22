''' xform_nodeps.py

This library contains the TimeConfigCore class from which the xform.TimeConfig class inherits
Almost all methods are defined here, but curcially not those which depend on Torch (hence the "nodeps")

This means you can define a much lighter docker image to do most of the needed transformations
'''

#~~~IMPORT~LIBRARIES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from typing import List, Tuple, Optional
from os import environ
import statistics
import datetime as DT
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~Exceptions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class GapFillIntegrityError(Exception):
    pass # this error is thrown when you have to gap fill too much

class StartDateError(Exception):
    # gap_fill_daily was given the wrong start_date
    def __init__(self, expected_date, found_date):
        print('StartDateError: expected {}, found {}'.format(expected_date, found_date))

class EndDateError(Exception):
    pass # gap_fill_daily was given the wrong start_date

class HistorySmallerThanSequence(Exception):
    # This exception is raised if you try to make a prediction with a less data history than the sequence length
    def __repr__(self):
        return 'It seems you are trying to assemble a sequence longer than the actual price history provided. That won\'t work.'

class SequenceWrongEndDate(Exception):
    def __repr__(self):
        return 'The sequence did not have the expected end_date'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




#~~~Dates~&~Market~Days~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# a lot of this code is NYSE centric
nyse = mcal.get_calendar('NYSE')

def plus_years(date: str, n_years: int) -> datetime:
    # given a date as 'YYYY-MM-DD', return the NEW date after adding/subtracting n_years
    new_date = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=int(n_years*365))
    return new_date.strftime('%Y-%m-%d')

def this_or_next_trading_day(date):
    # if date is a trading day, return it
    # if it is not, return the next trading day
    month_later = plus_years(date, 1/12)
    schedule = nyse.schedule(start_date=date, end_date=month_later)
    return schedule.index[0].strftime('%Y-%m-%d')

def n_trading_days_later(date, n):
    # given a trading day,
    # return the trading day n days in the future
    year_later = plus_years(date, 1)
    schedule = nyse.schedule(start_date=date, end_date=year_later)
    return schedule.index[n].strftime('%Y-%m-%d')
    
def trading_day_2_weeks_ago() -> str:
    # return a trading date about 2 weeks ago
    # note this should be MORE than the interval used in db.rs/full_stale_stocks
    today = DT.date.today()
    week_ago = today - DT.timedelta(days=15)
    schedule = nyse.schedule(start_date=week_ago, end_date=today)
    return schedule.index[0].strftime('%Y-%m-%d')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def to_jz(obj):
    try:
        return obj.to_jz()
    except AttributeError:
        pass 
    if type(obj).__name__ in ('int', 'float', 'str', 'bool'):
        return obj 
    elif type(obj).__name__ in ('list', 'tuple'):
        return [ to_jz(x) for x in obj ] 
    elif type(obj).__name__ == 'dict':
        return {  k:to_jz(v) for k,v in obj.items() }
    else:
        return { k:to_jz(v) for k,v in vars(obj).items() }
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DatePrice:
    # This class captures a price on a date, as well as indicating if interpolation had to be used
    # becuase the actual price was missing
    def __init__(self, date: str, price: float, interpolated: Optional[bool]=False):
        self.date: str = date 
        self.price: float = price 
        self.interpolated: bool = interpolated


class DateSigma:
    # This class captures a change in price between two days
    def __init__(self, start: DatePrice, end: DatePrice, pct_chg: float, sigma: float):
        self.start: DatePrice = start 
        self.end: DatePrice = end 
        self.pct_chg: float = pct_chg 
        self.sigma: float = sigma 
    
class SigmaHistory:
    def __init__(self, history: List[DateSigma], std_dev:float):
        self.history: List[DateSigma] = history 
        self.std_dev: float = std_dev

class TimeConfigCore:
    # The time config specifies an end_date, a number of years of history prior to that date, and a delta size in terms of trading days
    def __init__(self, 
            end_date: str,          # the last trading day to consider, i.e. '2022-11-31'
            years_history: int,     # the number of years of history of trading data you want to use, i.e. 10 etc
            delta_size: int,        # Calculate price changes between n consecutive business days 
            inc_offsets:Optional[bool]=True
            ):
        # construct a 'TimeConfig': given an end_date, the # of years of history you want, and the delta_size (in days)
        # verify integrity
        assert type(end_date).__name__ == 'str'                 # date should be provided in YYYY-MM-DD format
        assert type(delta_size).__name__ == 'int'
        assert delta_size >= 1 
        assert end_date == this_or_next_trading_day(end_date)   # ensure the user specified a valid trading day to end on
        # determine the start date and schedule
        start_date = this_or_next_trading_day(plus_years(end_date, -years_history))
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        # assign properties
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.years_history: int = years_history
        self.delta_size: int = delta_size        
        self.inc_offsets: bool = inc_offsets 
        self.trading_days: List[str] = [ date.strftime('%Y-%m-%d') for date in schedule.index ] # a list of the trading days for years_history ending at end_date 
        assert self.trading_days[0] == self.start_date 
        assert self.trading_days[-1] == self.end_date
        # determine the 'select dates' that will be bookends for delta pairs
        if inc_offsets == False:
            self.select_dates = [ self.trading_days[i] for i in range(len(self.trading_days)) if (len(self.trading_days) -i -1) % delta_size == 0 ]
            self.delta_pairs = [ (self.select_dates[i], self.select_dates[i+1]) for i in range(len(self.select_dates)-1) ]
        else:
            # When the TimeConfig struct was first introduced, each datapoint would cover a n-days
            # i.e. prices from Friday to Friday for 5 days
            # but why shouldn't you 'include offsets' to also have monday to monday prices etc. for more data?
            self.select_dates = self.trading_days[delta_size:]
            self.delta_pairs = [ (self.select_dates[i], self.select_dates[i+delta_size]) for i in range(len(self.select_dates)-delta_size) ]
        assert self.select_dates[-1] == self.end_date # this is critical for prediction time
        assert self.delta_pairs[-1][-1] == self.end_date

    def to_jz(self):
        # custom jsonification
        v = vars(self)
        fields = 'start_date end_date years_history delta_size inc_offsets'.split()
        return { k:v[k] for k in fields }
        

    def gap_fill_daily(self,
            daily_prices: List[Tuple[str, float]],  # A list of (date, price): i.e.  [('2022-03-15', 100), ('2022-03-16', 103), ('2022-03-18', 98), etc.]
            min_integrity:Optional[float]=0.95      # Throw an error if you have to fill gaps for more than 100*(1-min_integrity)% of the dates
            ) -> Tuple[List[Tuple[str, float]], List[DatePrice]]:
        # as long as daily_prices has the correct starting and ending days, interpolate any gaps
        # and by interpolate I mean assume nothing changed
        if daily_prices[0][0] != self.start_date:
            raise StartDateError(self.start_date, daily_prices[0][0] )
        if daily_prices[-1][0] != self.end_date:
            raise EndDateError
        if len(daily_prices) < min_integrity * len(self.trading_days):   
            raise GapFillIntegrityError # a few gaps are okay, but you should have most fo the data
        last_price = daily_prices[0][1]
        lookup = { k:v for k,v in daily_prices }
        contiguous_prices = []
        date_prices = []
        for date in self.trading_days:
            price = lookup.get(date, last_price)
            contiguous_prices.append((date, price))
            if date in lookup:
                date_prices.append(DatePrice(date, price))
            else:
                date_prices.append(DatePrice(date, price, interpolated=True))
            last_price = price
        return contiguous_prices, date_prices


    def sigmas(self, daily_prices: List[Tuple[str, float]], **kwargs):
        # given daily prices, convert it into a list of [start_date, end_date, pct_chg, sigma_level]
        # see also .sigma_history() which is similar
        contiguous_prices, date_prices = self.gap_fill_daily(daily_prices, **kwargs)
        lookup = { k:v for k,v in contiguous_prices }
        pair_changes, pct_changes = [], []
        for start_date, end_date in self.delta_pairs:
            start_price = lookup[start_date]
            end_price = lookup[end_date]
            pct_chg = 100*(end_price - start_price)/(start_price + 1e-5)
            pair_changes.append((start_date, end_date, pct_chg))
            pct_changes.append(pct_chg)
        x_stdev = statistics.stdev(pct_changes)
        sigmas = [ (x[0],x[1],x[2]/x_stdev) for x in pair_changes]
        return sigmas, x_stdev 
    

    def sigma_history(self, daily_prices: List[Tuple[str, float]], **kwargs) -> SigmaHistory:
        # given daily prices, convert it into a list of [DateSigma]
        # See also .sigmas() which is similar
        _, date_prices = self.gap_fill_daily(daily_prices, **kwargs)
        lookup = { dp.date:dp for dp in date_prices }
        proto = [] # proto results: you will need to divide by stdev
        for start_date, end_date in self.delta_pairs:
            start: DatePrice = lookup[start_date]
            end: DatePrice = lookup[end_date]
            pct_chg = 100*(end.price - start.price)/(start.price + 1e-5)
            proto.append((start, end, pct_chg))
        std_dev = statistics.stdev([ x[2] for x in proto ])
        history = [ DateSigma(start, end, pct_chg, pct_chg/std_dev) for start, end, pct_chg in proto ]
        return SigmaHistory(history, std_dev)


    def sequences(self,
        # combine successive sets of sigmas into sequences, returning [seq_start, seq_end, [sigma_levels] ]
        # this will be what you use for input to make predictions
            daily_prices: List[Tuple[str, float]], # A list of (date, price): i.e.  [('2022-03-15', 100), ('2022-03-16', 103), ('2022-03-18', 98), etc.]
            seq_length,     # Take windows of n successive sigma levels as input to your model
            **kwargs):
        sigmas, x_stdev = self.sigmas(daily_prices, **kwargs)
        if self.inc_offsets == False:
            sequences = [ (sigmas[i][0], sigmas[i+seq_length-1][1], [sigmas[i+j][2] for j in range(seq_length)]) for i in range(len(sigmas)-seq_length+1)]
        elif self.inc_offsets:
            sequences = [ (sigmas[i][0], sigmas[i+(seq_length-1)*(self.delta_size)][1], [sigmas[i+self.delta_size*sq][2] for sq in range(seq_length)]) for i in range(len(sigmas)-self.delta_size*(seq_length-1)) ]
        if sequences == []:
            raise HistorySmallerThanSequence
        if sequences[-1][1] != self.end_date:
            raise SequenceWrongEndDate
        return sequences, x_stdev 


    def xy_single(self, daily_prices, seq_length, prediction_deltas, **kwargs):
        # append percent change after the next prediction_deltas days to sequence data, removing those which don't have data
        # return a list of {seq_start, seq_end, pred_for, sigma_levels, pct_chg }
        if type(prediction_deltas).__name__ == 'int':
            prediction_deltas = [prediction_deltas] # prediction deltas should be a list of integers
        sequences, x_stdev = self.sequences(daily_prices, seq_length, **kwargs)
        contiguous_prices, date_prices = self.gap_fill_daily(daily_prices, **kwargs)
        lookup = { k:v for k,v in contiguous_prices} # date -> price
        xy_temp, changes_temp = [], []
        for seq_start, seq_end, x_sigmas in sequences:
            future_indexes = [ self.trading_days.index(seq_end) + days_ahead for days_ahead in prediction_deltas ]
            if prediction_deltas != []: 
                # prediction_deltas = [] at prediction time
                if max(future_indexes) > len(self.trading_days) - 1:
                    continue 
            future_dates = [ self.trading_days[f_idx] for f_idx in future_indexes ]
            future_prices = [ lookup[future_date] for future_date in future_dates ]
            seq_end_price = lookup[seq_end]
            future_pct_changes = [ 100*(future_price-seq_end_price)/seq_end_price for future_price in future_prices ]
            changes_temp.append(future_pct_changes)
            xy_temp.append((seq_start, seq_end, future_dates, x_sigmas, future_pct_changes))
        try:
            y_stdev = [ statistics.stdev([ future_pct_changes[di] for future_pct_changes in changes_temp]) for di in range(len(prediction_deltas)) ]
        except statistics.StatisticsError:
            y_stdev = [ 999 for di in range(len(prediction_deltas)) ] # mp.prediction_data() provides no y-values so there is nothing to take a standard deviation of
        xy_single = [ {'seq_start':seq_start, 'seq_end':seq_end, 'future_dates':future_date, 'x_seq_sigmas':sigmas, 'y_future_sigmas':[future_pct_changes[di]/(y_stdev[di]+1e-5) for di in range(len(future_pct_changes))]} for seq_start, seq_end, future_date, sigmas, future_pct_changes in xy_temp ]
        return xy_single, x_stdev, y_stdev


    def xy(self, input_tickers, output_tickers, prices_by_ticker, seq_length, prediction_deltas, **kwargs):
        # let prices_by_ticker be a dict of ticker->[(date,price),(date,price)] for all the tickers
        data = {}
        for ticker in input_tickers + output_tickers:
            daily_prices = prices_by_ticker[ticker]
            xy_single, x_stdev, y_stdev = self.xy_single(daily_prices, seq_length, prediction_deltas, **kwargs)
            data[ticker] = xy_single, x_stdev, y_stdev 
        # INTEGRITY VALIDATION: go back over the results for each ticker and ensure the timescales match
        for xy_single_1, _, y_stdev_1 in data.values():
            for xy_single_2, _, y_stdev_2 in data.values():
                for i in range(len(xy_single_1)):
                    for key in ['seq_start', 'seq_end', 'future_dates']:
                        assert xy_single_1[i][key] == xy_single_2[i][key]
                    for key in ['x_seq_sigmas', 'y_future_sigmas']:
                        assert len(xy_single_1[i][key]) == len(xy_single_2[i][key])
                assert len(y_stdev_1) == len(y_stdev_2)
        # note the order of comprehension: tickers first (they occur on the same date), sequence next (history at one point), then i (history at many points)
        xy = []
        for i in range(len(xy_single_1)): # xy_single_1 is arbitrary, they all have the same dates as verified above
            dp = {} # one datapoint
            for key in ['seq_start', 'seq_end', 'future_dates']:
                dp[key] = xy_single_1[i][key]
            dp['x_seq_sigmas'] = [ [  data[ticker][0][i]['x_seq_sigmas'][j] for ticker in input_tickers ] for j in range(seq_length) ]
            dp['y_future_sigmas'] = [ data[ticker][0][i]['y_future_sigmas'][di] for ticker in output_tickers for di in range(len(prediction_deltas)) ] # flatten
            xy.append(dp)
        x_stdevs = [ data[ticker][1] for ticker in input_tickers ]
        y_stdevs = [ data[ticker][2][di] for ticker in output_tickers for di in range(len(prediction_deltas)) ]
        return xy, x_stdevs, y_stdevs


    def split_xy(self, xy, test_start=None, test_end=None):
        # given an xy output from above, split it into test and training sets
        # BEFORE 2022.05.27: by defaut, the cv set is a 'heart cut' out of the middle of dataset,
        #     with a buffer on either side to ensure no single day is shared by both a training sequence and a CV sequence
        # ON 2022.05.27 - I realized models predicted better a few weeks after their training end_date but not as well later
        # This implies they don't have enough recency bias- I dropped the heart cut idea to get enough CV data with a shorter years_history
        # CHANGED 2023.01.12: test_start 0.0->0.85, test_end -> 1.0
        if not test_start:
            test_start = 0.85
        if not test_end:
            test_end = 1.0 
        assert test_start < test_end
        cv_start_date = self.trading_days[int(test_start*len(self.trading_days))]
        cv_end_date = self.trading_days[min(int(test_end*len(self.trading_days)), len(self.trading_days)-1)]
        xy_train = [ dp for dp in xy if (dp['seq_end'] <= cv_start_date or dp['seq_start'] >= cv_end_date ) ]
        xy_cv = [ dp for dp in xy if (dp['seq_start'] > cv_start_date and dp['seq_end'] < cv_end_date ) ]
        buffer_size = len(xy) - len(xy_train) - len(xy_cv)
        tot = len(xy)
        if environ.get('SHOW_SPLITS', '').lower() in ('true', '1', 'y'):
            print("   Split {:,} datapoints into {:,} train, {:,} CV, {:,} buffer".format(tot, len(xy_train), len(xy_cv), buffer_size))
        return xy_train, xy_cv, cv_start_date, cv_end_date
        


def decompose_y_stdevs(y_stdevs, output_tickers, prediction_deltas):
    ''' making predictions with the y_stdevs from the output of tc.xy(...) necessitates
    "decomposing" it into the terms for each ticker/days ahead. 
    This function performs that decomposition. '''
    y_stdev_decomp = { (arr_index, output_tickers[arr_index], prediction_deltas[di]):y_stdevs[len(prediction_deltas)*arr_index + di] for arr_index in range(len(output_tickers)) for di in range(len(prediction_deltas)) }
    return y_stdev_decomp


def price_changes(prices, days_ahead):
    # given a list of price changes, return a list days_ahead shorter
    # that captures the percent change over successive days_ahead intervals
    pct_changes = []
    for i in range(days_ahead, len(prices)):
        delta = prices[i]-prices[i-days_ahead]
        pct = 100*delta/(prices[i-days_ahead]+0.001)
        pct_changes.append(pct)
    return pct_changes



def test_tc_delta_pairs():
    # The .delta_pairs property of TimeConfig was very difficult to get right
    # inspecting this test is a good way to understand what .delta_pairs means
    tc2 = TimeConfigCore('2022-03-24', 2.1/52, 2)
    assert tc2.select_dates == ['2022-03-10', '2022-03-14', '2022-03-16', '2022-03-18', '2022-03-22', '2022-03-24']
    assert tc2.delta_pairs == [('2022-03-10', '2022-03-14'), ('2022-03-14', '2022-03-16'),
        ('2022-03-16', '2022-03-18'), ('2022-03-18', '2022-03-22'), ('2022-03-22', '2022-03-24')]
    tc3 = TimeConfigCore('2022-03-24', 2.1/52, 3)
    assert tc3.select_dates == ['2022-03-11', '2022-03-16', '2022-03-21', '2022-03-24']
    assert tc3.delta_pairs == [('2022-03-11', '2022-03-16'), ('2022-03-16', '2022-03-21'), ('2022-03-21', '2022-03-24')]
    tc4 = TimeConfigCore('2022-03-24', 2.1/52, 4)
    assert tc4.select_dates == ['2022-03-14', '2022-03-18', '2022-03-24']
    assert tc4.delta_pairs == [('2022-03-14', '2022-03-18'), ('2022-03-18', '2022-03-24')]
    assert tc2.trading_days == tc3.trading_days == tc4.trading_days




def test_tc_gap_fill():
    # ensure the gap fill function fills gaps correctly
    tc = TimeConfigCore('2022-03-24', 1.4/52, 2)
    daily_prices = [('2022-03-15', 100), ('2022-03-16', 103), ('2022-03-18', 98), ('2022-03-21',101), ('2022-03-22',107), ('2022-03-24',110)]
    contiguous_prices, date_prices = tc.gap_fill_daily(daily_prices, min_integrity=0.5)
    assert contiguous_prices == [('2022-03-15', 100), ('2022-03-16', 103), ('2022-03-17', 103), ('2022-03-18', 98), ('2022-03-21', 101),
        ('2022-03-22', 107), ('2022-03-23', 107), ('2022-03-24', 110)]


def test_tc_sigma_sequence():
    # ensure getting sigmas and sequences does what you want it to 
    # inspecting this test is a good way to understand what the .sigmas() and .sequences() methods do 
    tc = TimeConfigCore('2022-03-24', 2.5/52, 3)
    daily_prices = [('2022-03-07', 100), ('2022-03-10', 88), ('2022-03-11', 95), ('2022-03-15', 100), ('2022-03-16', 103),
        ('2022-03-18', 98), ('2022-03-21',101), ('2022-03-22',107), ('2022-03-24',110)]
    assert tc.gap_fill_daily(daily_prices, min_integrity=0.6) == [
        ('2022-03-07', 100), ('2022-03-08', 100), ('2022-03-09', 100), ('2022-03-10', 88), ('2022-03-11', 95),
        ('2022-03-14', 95), ('2022-03-15', 100), ('2022-03-16', 103), ('2022-03-17', 103), ('2022-03-18', 98),
        ('2022-03-21', 101), ('2022-03-22', 107), ('2022-03-23', 107), ('2022-03-24', 110)]
    assert tc.sigmas(daily_prices, min_integrity=0.6) == (
        [('2022-03-08', '2022-03-11', -0.7022096842611291), ('2022-03-11', '2022-03-16', 1.1826689356889077),
        ('2022-03-16', '2022-03-21', -0.2727027907986018), ('2022-03-21', '2022-03-24', 1.2514628048727672)],
        7.120379584712068)
    assert tc.sequences(daily_prices, 2, min_integrity=0.6) == (
        [('2022-03-08', '2022-03-16', [-0.7022096842611291, 1.1826689356889077]),
        ('2022-03-11', '2022-03-21', [1.1826689356889077, -0.2727027907986018]),
        ('2022-03-16', '2022-03-24', [-0.2727027907986018, 1.2514628048727672])],
        7.120379584712068)
    assert tc.sequences(daily_prices, 3, min_integrity=0.6) == (
        [('2022-03-08', '2022-03-21', [-0.7022096842611291, 1.1826689356889077, -0.2727027907986018]),
        ('2022-03-11', '2022-03-24', [1.1826689356889077, -0.2727027907986018, 1.2514628048727672])],
        7.120379584712068)

def test_sequences_inc_offsets():
    # getting sigmas right with offsets was NOT easy!
    delta_size = 2
    seq_length = 5
    tc25 = TimeConfigCore('2022-04-14', 5.5/52, delta_size, inc_offsets=True)
    daily_prices = [('2022-03-07', 99), ('2022-03-10', 88), ('2022-03-11', 95), ('2022-03-15', 100), ('2022-03-16', 103),
        ('2022-03-18', 98), ('2022-03-21',101), ('2022-03-22',107), ('2022-03-24',110),
        ('2022-03-28', 112), ('2022-03-29', 114), ('2022-03-31', 110), ('2022-04-04', 105),
        ('2022-04-05', 103), ('2022-04-06', 108), ('2022-04-08', 113), ('2022-04-11', 120),
        ('2022-04-12', 105), ('2022-04-14', 105)]
    sigmas, _ = tc25.sigmas(daily_prices, min_integrity=0.4)
    assert sigmas == [('2022-03-09', '2022-03-11', -0.6964948353879606), ('2022-03-10', '2022-03-14', 1.3712241898566124),
        ('2022-03-11', '2022-03-15', 0.9072761632914098), ('2022-03-14', '2022-03-16', 1.4516418612662558),
        ('2022-03-15', '2022-03-17', 0.5171474157979318), ('2022-03-16', '2022-03-18', -0.8368081186151881),
        ('2022-03-17', '2022-03-21', -0.33472324744607523), ('2022-03-18', '2022-03-22', 1.5831043308444768),
        ('2022-03-21', '2022-03-23', 1.0240542897226907), ('2022-03-22', '2022-03-24', 0.4833153450291018),
        ('2022-03-23', '2022-03-25', 0.4833153450291018), ('2022-03-24', '2022-03-28', 0.31342267909046756),
        ('2022-03-25', '2022-03-29', 0.6268453581809351), ('2022-03-28', '2022-03-30', 0.30782584603499785),
        ('2022-03-29', '2022-03-31', -0.6048507861390227), ('2022-03-30', '2022-04-01', -0.6048507861390227),
        ('2022-03-31', '2022-04-04', -0.7835566977261689), ('2022-04-01', '2022-04-05', -1.0969793768166365),
        ('2022-04-04', '2022-04-06', 0.4925213507243222), ('2022-04-05', '2022-04-07', 0.8368081186151881),
        ('2022-04-06', '2022-04-08', 0.7980670055997732), ('2022-04-07', '2022-04-11', 1.9153608134394557),
        ('2022-04-08', '2022-04-12', -1.22040689497915), ('2022-04-11', '2022-04-13', -2.154780935071061),
        ('2022-04-12', '2022-04-14', 0.0)]
    # inspect the results below to gain a better understanding of the the sequences method works for inc_offsets=True
    sequences, _ = tc25.sequences(daily_prices, seq_length,  min_integrity=0.4)
    assert sequences[0]  == ('2022-03-09', '2022-03-23', [-0.6964948353879606, 0.9072761632914098, 0.5171474157979318, -0.33472324744607523, 1.0240542897226907])
    assert sequences[1]  == ('2022-03-10', '2022-03-24', [1.3712241898566124, 1.4516418612662558, -0.8368081186151881, 1.5831043308444768, 0.4833153450291018])
    assert sequences[-1] == ('2022-03-31', '2022-04-14', [-0.7835566977261689, 0.4925213507243222, 0.7980670055997732, -1.22040689497915, 0.0])
    # Repeat similar tests for different delta_size, seq_length
    delta_size = 4
    seq_length = 2
    tc42 = TimeConfigCore('2022-04-14', 5.5/52, delta_size, inc_offsets=True)
    sigmas, _ = tc42.sigmas(daily_prices, min_integrity=0.4)
    assert sigmas == [('2022-03-11', '2022-03-17', 1.2662177986129162), ('2022-03-14', '2022-03-18', 0.47483167447984354),
        ('2022-03-15', '2022-03-21', 0.15036336437666983), ('2022-03-16', '2022-03-22', 0.5839353973383353),
        ('2022-03-17', '2022-03-23', 0.5839353973383353), ('2022-03-18', '2022-03-24', 1.8411840498343572), 
        ('2022-03-21', '2022-03-25', 1.3398715650791089), ('2022-03-22', '2022-03-28', 0.7026325484128751), 
        ('2022-03-23', '2022-03-29', 0.9836855677780253), ('2022-03-24', '2022-03-30', 0.5467758754313068), 
        ('2022-03-25', '2022-03-31', 0.0), ('2022-03-28', '2022-04-01', -0.26850601069233165), 
        ('2022-03-29', '2022-04-04', -1.1870792070255576), ('2022-03-30', '2022-04-05', -1.4508745863645705), 
        ('2022-03-31', '2022-04-06', -0.2733879377156534), ('2022-04-01', '2022-04-07', -0.2733879377156534), 
        ('2022-04-04', '2022-04-08', 1.1456256388014154), ('2022-04-05', '2022-04-11', 2.481725438687925), 
        ('2022-04-06', '2022-04-12', -0.4176760152513123), ('2022-04-07', '2022-04-13', -0.4176760152513123), 
        ('2022-04-08', '2022-04-14', -1.064519406063814)]
    sequences, _ = tc42.sequences(daily_prices, seq_length,  min_integrity=0.4)
    assert sequences[0]  == ('2022-03-11', '2022-03-23', [1.2662177986129162, 0.5839353973383353])
    assert sequences[1]  == ('2022-03-14', '2022-03-24', [0.47483167447984354, 1.8411840498343572])
    assert sequences[-1] == ('2022-04-04', '2022-04-14', [1.1456256388014154, -1.064519406063814])


def test_tc_xy_single():
    # test the xy_single function that shows future price changes for a single ticker
    tc = TimeConfigCore('2022-03-24', 2.5/52, 3)
    daily_prices = [('2022-03-07', 100), ('2022-03-10', 88), ('2022-03-11', 95), ('2022-03-15', 100), 
        ('2022-03-16', 103), ('2022-03-18', 98), ('2022-03-21',101), ('2022-03-22',107), ('2022-03-24',110)]
    assert tc.gap_fill_daily(daily_prices, min_integrity=0.6) == [
        ('2022-03-07', 100), ('2022-03-08', 100), ('2022-03-09', 100), ('2022-03-10', 88), ('2022-03-11', 95),
        ('2022-03-14', 95), ('2022-03-15', 100), ('2022-03-16', 103), ('2022-03-17', 103), ('2022-03-18', 98),
        ('2022-03-21', 101), ('2022-03-22', 107), ('2022-03-23', 107), ('2022-03-24', 110)]
    assert tc.sequences(daily_prices, 2, min_integrity=0.6) == (
        [('2022-03-08', '2022-03-16', [-0.7022096842611291, 1.1826689356889077]),
        ('2022-03-11', '2022-03-21', [1.1826689356889077, -0.2727027907986018]), 
        ('2022-03-16', '2022-03-24', [-0.2727027907986018, 1.2514628048727672])], 
        7.120379584712068)
    assert tc.xy_single(daily_prices, 2, [0,1,2], min_integrity=0.6) == (
        [{'seq_start': '2022-03-08', 'seq_end': '2022-03-16', 'future_dates': ['2022-03-16', '2022-03-17', '2022-03-18'],
            'x_seq_sigmas': [-0.7022096842611291, 1.1826689356889077], 'y_future_sigmas': [0.0, 0.0, -0.6359545087961614]},
        {'seq_start': '2022-03-11', 'seq_end': '2022-03-21', 'future_dates': ['2022-03-21', '2022-03-22', '2022-03-23'],
            'x_seq_sigmas': [1.1826689356889077, -0.2727027907986018], 'y_future_sigmas': [0.0, 1.4142101957144433, 0.7782572008634214]}],
        7.120379584712068, [0.0, 4.20063434368246, 7.633191533908419])


def test_tc_xy():
    # test the xy_single function that shows future price changes for multiple inputs and outputs
    # inspecting this test and comparing to test_tc_xy_single is helpful in understanding the xy() method
    tc = TimeConfigCore('2022-03-24', 2.5/52, 3)
    pricesA = [('2022-03-07', 100), ('2022-03-10',  88), ('2022-03-11',  95), ('2022-03-15', 100), ('2022-03-16', 103), ('2022-03-18',  98), ('2022-03-21', 101), ('2022-03-22',107), ('2022-03-24',110)]
    pricesB = [('2022-03-07', 205), ('2022-03-10', 180), ('2022-03-11', 190), ('2022-03-15', 200), ('2022-03-16', 195), ('2022-03-18', 198), ('2022-03-21', 190), ('2022-03-22',188), ('2022-03-24',180)]
    pricesC = [('2022-03-07',  22), ('2022-03-10',  23), ('2022-03-11',  20), ('2022-03-15',  22), ('2022-03-16',  19), ('2022-03-18',  21), ('2022-03-21',  25), ('2022-03-22', 27), ('2022-03-24', 28)]
    input_tickers = ['A', 'B', 'C']
    output_tickers = ['A', 'B']
    prices_by_ticker = {'A':pricesA, 'B':pricesB, 'C':pricesC}
    xy, x_stdevs, y_stdevs = tc.xy(input_tickers, output_tickers, prices_by_ticker, 2, [0,1,2], min_integrity=0.6)
    assert xy == [
        {'seq_start': '2022-03-08', 'seq_end': '2022-03-16', 'future_dates': ['2022-03-16', '2022-03-17', '2022-03-18'],
            'x_seq_sigmas': [[-0.7022096842611291, -1.6996855634438666, -0.49027052201379123], [1.1826689356889077, 0.6112904195862029, -0.2696487748508283]],
            'y_future_sigmas': [0.0, 0.0, -0.6359545087961614, 0.0, 0.0, 0.8396847196762266]},
        {'seq_start': '2022-03-11', 'seq_end': '2022-03-21', 'future_dates': ['2022-03-21', '2022-03-22', '2022-03-23'],
            'x_seq_sigmas': [[1.1826689356889077, 0.6112904195862029, -0.2696487748508283], [-0.2727027907986018, -0.5956163070672805, 1.7030448489777577]],
            'y_future_sigmas': [0.0, 1.4142101957144433, 0.7782572008634214, 0.0, -1.4141945626283574, -0.574521123988997]}]
    assert x_stdevs == [7.120379584712068, 4.304956735041998, 18.5426301408834]
    assert y_stdevs == [0.0, 4.20063434368246, 7.633191533908419, 0.0, 0.7443229275647868, 1.8321795140056292]


def test_decompose_y_stdevs():
    # test the decompose_y_stdevs function
    y_stdevs = [0.0, 4.20063434368246, 7.633191533908419, 0.0, 0.7443229275647868, 1.8321795140056292]
    output_tickers = ['A', 'B']
    prediction_deltas = [0,1,2]
    y_stdev_decomp = decompose_y_stdevs(y_stdevs, output_tickers, prediction_deltas)
    assert  y_stdev_decomp == {
        (0, 'A', 0): 0.0, (0, 'A', 1): 4.20063434368246, (0, 'A', 2): 7.633191533908419,
        (1, 'B', 0): 0.0, (1, 'B', 1): 0.7443229275647868, (1, 'B', 2): 1.8321795140056292} 

