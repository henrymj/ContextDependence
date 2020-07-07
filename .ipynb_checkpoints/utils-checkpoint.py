import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import matplotlib.patches as patches
import numpy as np
from os import path, makedirs
from glob import glob
from cycler import cycler
from matplotlib.ticker import FormatStrFormatter
from collections import Counter
import statistics      

# IMPORTS FOR FIGURE 2C
import math
import scipy.stats as stats
import rpy2
from rpy2.robjects.packages import importr
try:
    BayesFactor = importr('BayesFactor')
except:
    rutils = importr('utils')
    rutils.install_packages('BayesFactor', repo="http://cran.rstudio.com/") #ONLY NEED TO DO THIS ONCE TO INSTALL BAYESFACTOR

########################################################
####################### CONSTANTS ######################
########################################################

class formats:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# paths to various output directories
DATA_DIR = 'Data'
SUMMARY_DIR = 'SummaryData'
FIGURE_DIR = 'Figures'
SUPPLEMENT_DIR = path.join(FIGURE_DIR, 'SupplementaryFigures')
S6_DIR = path.join(SUPPLEMENT_DIR, 'S6')


for directory in [SUMMARY_DIR, FIGURE_DIR, SUPPLEMENT_DIR, S6_DIR]:
    makedirs(directory, exist_ok=True)
    


# Dictionary for easier reading
verbose_names ='''Fixed SSDs 1
Fixed SSDs 2
Deadline 1 300ms
Deadline 1 500ms
Deadline 1 700ms
Deadline 2 300ms
Deadline 2 500ms
Deadline 2 700ms
Stop Probability .2
Stop Probabllity .4
Saccadic Eye Movements
Between-Subjects Modality Auditory 1
Between-Subjects Modality Auditory 2
Between-Subjects Modality Visual 1
Between-Subjects Modality Visual 2
Between-Subjects Stimulus Selective Stop
Within-Subjects Central Go Simple Stop
Within-Subjects Central Go Selective Stop
Within-Subjects Peripheral Go Simple Stop
Within-Subjects Peripheral Go Selective Stop
Turk Simple .2
Turk Simple .4
Turk Stim Selective
Turk Motor Selective
Variable Difficulty'''.split('\n')

short_names = '''FixedSSDs1
FixedSSDs2
Deadline1300ms
Deadline1500ms
Deadline1700ms
Deadline2300ms
Deadline2500ms
Deadline2700ms
StopProbabilityLow
StopProbabilityHigh
Saccades
BtwnSubjAuditory1
BtwnSubjAuditory2
BtwnSubjVisual1
BtwnSubjVisual2
BtwnSubjStimSelec
WithinSubjCentralGoSimple
WithinSubjCentralGoSelec
WithinSubjPeriphGoSimple
WithinSubjPeriphGoSelec
TurkSimpleLow
TurkSimpleHigh
TurkStimSelec
TurkMotorSelec
Matzke'''.split('\n')


condition_name_dict = {}
for short, long in zip(short_names, verbose_names):
    condition_name_dict[short] = long

#these are the 339 subjects who passed the criteria discussed in the manuscriptinfo
full_passed_turkers = [4, 5, 9, 11, 12, 13, 15, 16, 17, 19, 20, 23, 25, 28, 30, 31, 32,
               33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 51,
               52, 53, 54, 55, 56, 58, 59, 60, 62, 63, 64, 66, 71, 74, 76, 80,
               81, 82, 85, 87, 88, 89, 90, 91, 92, 94, 96, 97, 99, 100, 101,
               102, 103, 105, 106, 107, 108, 109, 112, 113, 114, 117, 122, 123,
               124, 125, 126, 127, 128, 132, 133, 134, 136, 139, 140, 141, 142,
               144, 146, 150, 151, 154, 156, 158, 160, 161, 162, 165, 167, 168,
               171, 172, 173, 175, 177, 179, 180, 183, 184, 185, 186, 188, 189,
               190, 193, 195, 198, 199, 201, 202, 203, 205, 207, 208, 210, 212,
               214, 215, 216, 217, 219, 220, 221, 222, 223, 226, 227, 228, 230,
               232, 233, 234, 235, 236, 237, 239, 240, 241, 243, 244, 245, 246,
               247, 251, 252, 254, 256, 257, 259, 260, 266, 268, 270, 271, 273,
               274, 276, 278, 280, 282, 283, 284, 285, 288, 292, 293, 294, 295,
               296, 297, 298, 299, 301, 302, 304, 305, 307, 309, 311, 313, 314,
               317, 323, 324, 325, 327, 328, 330, 333, 334, 335, 336, 338, 340,
               342, 343, 345, 346, 349, 350, 354, 355, 357, 358, 360, 362, 363,
               364, 365, 368, 369, 375, 376, 377, 383, 384, 386, 387, 388, 389,
               390, 394, 396, 397, 398, 399, 400, 403, 405, 406, 407, 409, 410,
               411, 412, 414, 415, 416, 418, 419, 420, 421, 422, 423, 425, 426,
               427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 438, 439, 441,
               443, 444, 445, 446, 448, 449, 450, 451, 454, 456, 457, 458, 459,
               460, 461, 463, 465, 466, 467, 469, 470, 473, 474, 476, 477, 479,
               480, 481, 482, 483, 486, 489, 492, 493, 494, 495, 496, 497, 499,
               501, 504, 505, 507, 508, 509, 512, 514, 515, 516, 520, 521, 522,
               523, 526, 527, 529, 533, 535, 539, 543, 545, 546, 548, 549, 550,
               559, 560]
    



########################################################
######################## HELPERS #######################
########################################################
    
def read_cond_file(filey):
    cond_df = pd.read_excel(filey)
    cond_df = cond_df.replace(r'^\s*$', np.nan, regex=True).replace('?', np.nan)  
    return cond_df

def get_attr(rout, attr='bf'):
    # Takes in an rpy2 vector, returns the value of an attribute
    # Used for BayesFactor
        try:
            index = list(rout.names).index(attr)
            val = list(rout.items())[index][1]
            if len(val) == 1:
                val = val[0]
            if type(val)==rpy2.robjects.vectors.Matrix:
                val = np.asarray(val)
            return val
        except ValueError:
            print('Did not pass a valid attribute')
            

########################################################
################# VIOLATION ANLYSIS ####################
########################################################

def violation_analysis(d, save_results=False, verbose=False):
    
    condition_name = 'UNKNOWN'
    if isinstance(d, str): #if it's a string, read in the file and print the condition name (used to v_a.ipynb)
        condition_name = d.replace('Data/Data', '').replace('.xlsx', '')#drop path and extension.
        if verbose: print(formats.BOLD + '*'*80 + formats.END)
        if verbose: print(formats.BOLD + condition_name + formats.END)
        d = read_cond_file(d)
        
    
    if 'GoCriticalRT' in d.columns: 
        go_key = 'GoCriticalRT'
    elif 'Target.RT' in d.columns: #ADDED FOR MATZKE DATA
        go_key = 'Target.RT'
    else: 
        go_key = 'GoRT'
    
        
    stopFailRT = 'StopFailureRT'
    if 'TargetDelay.RT' in d.columns: stopFailRT = 'TargetDelay.RT' #ADDED FOR MATZKE DATA
        
    
    if 'Block' not in d.columns: d['Block'] = 1 #ADDED FOR MATZKE DATA - insert dummy block
    
    info = []
    condition_subjects = d.Subject.unique()
    if len(condition_subjects)==522: condition_subjects = full_passed_turkers
    for subject in condition_subjects:
        subdata = d.query('Subject == %d' % subject)
        if verbose: print('subject %d: found %d trials' % (subject, subdata.shape[0]))
        ssdvals = [i for i in subdata.StopSignalDelay.unique() if isinstance(i, float) and i >=0] #ignore nan/missing values and 0/negative SSDs
        ssdvals.sort()

        # You find all pairs of trials in which the first is a go trials with a response and the second is a 
        # stop trial with a response. Both trials need to come from the same subject and block. 
        # This should be done separately for each SSD. Then the RT for the first trial in the pair should be 
        # subtracted from the second. This is the core analysis per SSD per subject.

        for ssd in ssdvals:
            ssd_data = subdata.query('StopSignalDelay == %d' % ssd)
            signal_respond_data = ssd_data.dropna(subset=[stopFailRT])
            signal_respond_data = signal_respond_data.loc[signal_respond_data[stopFailRT] > 0, :]
            # for each signal respond trial, determine whether the previous trial was a go trial in the same block
            signal_respond_data['MatchingGo'] = np.nan
            if signal_respond_data.shape[0] > 0:
                for t in signal_respond_data.index:
                    if t < 1:
                        continue
                    prevtrial = d.loc[int(t) - 1, :]
                    if prevtrial.Block == signal_respond_data.loc[t, 'Block']: # blockmatch
                        if (prevtrial[go_key] > 0.0): #skip omissions
                            signal_respond_data.loc[t, 'MatchingGo'] = prevtrial[go_key]
                signal_respond_data = signal_respond_data.dropna(subset=['MatchingGo'])
            # I expect to see RuntimeWarnings when there only nans passed to nanmean (i.e. no stop failures and/or preceding GoRTs at some ssd)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                info.append([subject, 
                             ssd, 
                             ssd_data.shape[0], 
                             signal_respond_data.shape[0],
                             np.nanmean(signal_respond_data[stopFailRT]\
                                          - signal_respond_data.MatchingGo),
                             np.nanmean(signal_respond_data[stopFailRT]),
                             np.nanmean(signal_respond_data.MatchingGo)])

    info_df = pd.DataFrame(info, columns=['subject', 'ssd', 'nstoptrials', 
                                             'n_matched_go', 'mean_violation', 'mean_stopFailureRT', 'mean_precedingGoRT'])


    # However, I only include data at a given SSD for a given subject
    # (for things like figures or inferential stats) if there are at least 2 pairs of trials for that subject. 

    info_df = info_df.query('n_matched_go > 1')

    # And I only include an SSD for the experiment (for things like Figure 1) if 
    # 5 subjects have at least 2 pairs of trials for that SSD. So to reproduce the main data, 
    # you'll want to apply these same thresholds.

    all_ssdvals = info_df.ssd.unique()
    all_ssdvals.sort()

    for ssd in all_ssdvals:
        info_df_ssd = info_df.query('ssd == %d' % ssd)
        if verbose: print(ssd, 'n subs with data:', info_df_ssd.shape[0])
        if info_df_ssd.shape[0] < 5:
            info_df = info_df.query('ssd != %d' % ssd)
            if verbose: print('dropping', ssd)

    #Save summary data
    if save_results: info_df.to_csv(path.join(SUMMARY_DIR, f'summarydata{condition_name}.csv'), index=False)
    
    return info_df
    
    
########################################################
#################### SSRT FUNCTIONS ####################
########################################################

def calc_SSRT_wOmission(subj_df):
    
    #get TrialType column name
    trialType_col = 'TrialType'
    if 'TrialType' not in subj_df.columns: trialType_col='SS' #ADDED FOR MATZKE DATA
    
    # get SSD column name
    SSD_col = 'StopSignalDelay'
    if 'StopSignalDelay' not in subj_df.columns: SSD_col = 'SSD' #ADDED FOR MATZKE DATA
        
    #Set up keys for stop trial types
    stopTrialKey = 'stop'
    if 'Stop' in subj_df[trialType_col].unique(): stopTrialKey = 'Stop' #ADDED FOR SACCADE TASK
        
    #setting up stop failure RT key
    stopFailRT = 'StopFailureRT'
    if 'TargetDelay.RT' in subj_df.columns: stopFailRT = 'TargetDelay.RT' #ADDED FOR MATZKE DATA
    
    #Set up keys for go trial types
    if 'GoCritical' in subj_df[trialType_col].unique(): #ADDED FOR TURK MOTOR SELECTIVE STOP
        goTrialKey = 'GoCritical'
        goRT_key = 'GoCriticalRT'
    elif 'Target.RT' in subj_df.columns: #ADDED FOR MATZKE DATA
        goTrialKey = 'go'
        goRT_key = 'Target.RT'
    else: 
        goTrialKey = 'go'
        goRT_key = 'GoRT'


    ################################################
    # Getting MaxRT, P(respond|Signal), mean_SSD #, omission #s

    #get the mean SSD to subtract from the nth RT
    mean_SSD = subj_df[SSD_col].mean()

    # Calculate P(respond | signal) as num_stop_failures / num_stop_trials
    num_stop_failures = subj_df.loc[(subj_df[trialType_col]==stopTrialKey) & (subj_df[stopFailRT]>0), stopFailRT].count()
    num_stop_trials = len(subj_df.loc[subj_df[trialType_col]==stopTrialKey, stopFailRT])
    P_respond = num_stop_failures / num_stop_trials
    
    if (P_respond==0.0) | (P_respond==1.0):
        return np.nan, mean_SSD

    #get the omission count and the omission rate
    goRTs = subj_df.loc[(subj_df[trialType_col]==goTrialKey) & (subj_df[goRT_key]>0), goRT_key].values
    goRTs.sort()
    num_go_trials = subj_df.loc[subj_df[trialType_col]==goTrialKey, trialType_col].count()
    num_go_responses = len(goRTs)

    omission_count = num_go_trials - num_go_responses
    omission_rate = omission_count/num_go_trials

    #Correct P(respond | signal)
    corrected_P_respond = P_respond/(1-omission_rate)

    #get nth RT using corrected P(respond | signal)
    nth_index = int(np.rint(P_respond*len(goRTs))) - 1
    if nth_index < 0:
        nth_RT = goRTs[0]
    elif nth_index >= len(goRTs):
        nth_RT = goRTs[-1]
    else:
        nth_RT = goRTs[nth_index]

    ################################################
    # Calculate SSRT as nth_RT - mean_SSD
    SSRT = nth_RT - mean_SSD

    return SSRT, mean_SSD

def calc_SSRT_wReplacement(subj_df, maxRT=None):
    
    #get TrialType column name
    trialType_col = 'TrialType'
    if 'TrialType' not in subj_df.columns: trialType_col='SS' #ADDED FOR MATZKE DATA
    
    # get SSD column name
    SSD_col = 'StopSignalDelay'
    if 'StopSignalDelay' not in subj_df.columns: SSD_col = 'SSD' #ADDED FOR MATZKE DATA
        
    #Set up keys for stop trial types
    stopTrialKey = 'stop'
    if 'Stop' in subj_df[trialType_col].unique(): stopTrialKey = 'Stop' #ADDED FOR SACCADE TASK
        
    #setting up stop failure RT key
    stopFailRT = 'StopFailureRT'
    if 'TargetDelay.RT' in subj_df.columns: stopFailRT = 'TargetDelay.RT' #ADDED FOR MATZKE DATA
    
    #Set up keys for go trial types
    if 'GoCritical' in subj_df[trialType_col].unique(): #ADDED FOR TURK MOTOR SELECTIVE STOP
        goTrialKey = 'GoCritical'
        goRT_key = 'GoCriticalRT'
    elif 'Target.RT' in subj_df.columns: #ADDED FOR MATZKE DATA
        goTrialKey = 'go'
        goRT_key = 'Target.RT'
    else: 
        goTrialKey = 'go'
        goRT_key = 'GoRT'
    
    ################################################
    # Getting MaxRT, P(respond|Signal), mean_SSD #
    
    #If condition's maxRT is not passed in, get subject's
    if maxRT is None:
        maxRT = subj_df.max()[goRT_key]
    
    #get the mean SSD to subtract from the nth RT
    mean_SSD = subj_df[SSD_col].mean()

    # Calculate P(respond | signal) as num_stop_failures / num_stop_trials
    num_stop_failures = subj_df.loc[(subj_df[trialType_col]==stopTrialKey) & (subj_df[stopFailRT]>0), stopFailRT].count()
    num_stop_trials = len(subj_df.loc[subj_df[trialType_col]==stopTrialKey, stopFailRT])
    P_respond = num_stop_failures / num_stop_trials

    if (P_respond==0.0) | (P_respond==1.0):
        return np.nan, mean_SSD

    ################################################
    ############## Getting Nth RT ##################


    # 1 - create index of RTs to include 
    subj_df['include_for_SSRT'] = 0
    subj_df.loc[subj_df[trialType_col]==goTrialKey, 'include_for_SSRT'] = 1 #add all go trials 

    # 2 - get values for each RT being included
    subj_df['RTs_for_SSRT'] = np.nan
    # add in all GoRTs
    goRT_df = subj_df.loc[subj_df.include_for_SSRT==1, goRT_key].copy() #subset to the goRT columns
    goRTs = goRT_df.values # get the nonNaN from each row
    subj_df.loc[subj_df.include_for_SSRT==1, 'RTs_for_SSRT'] = goRTs 

    # fill go ommissions (and fast stopFailureRTs) with max GoRt
    go_omission_idx = (subj_df.include_for_SSRT==1) & (subj_df.RTs_for_SSRT.isnull())
    if len(subj_df.loc[go_omission_idx]) > 0:
        subj_df.loc[go_omission_idx, 'RTs_for_SSRT'] = maxRT

    # 3 - get Nth RT
    # get RTs used for SSRT and sort 
    useful_RTs = subj_df.RTs_for_SSRT.values.copy()
    useful_RTs = [i for i in useful_RTs if i==i]
    useful_RTs.sort()
    #Get the number of RTs to be used
    num_RTs = len(useful_RTs)

    # Get the nth_RT based on P(respond | Signal)
    nth_index = int(np.rint(P_respond*len(useful_RTs))) - 1
    if nth_index < 0:
        nth_RT = useful_RTs[0]
    elif nth_index >= len(goRTs):
        nth_RT = useful_RTs[-1]
    else:
        nth_RT = useful_RTs[nth_index]
    ################################################

    # Calculate SSRT as nth_RT - mean_SSD
    SSRT = nth_RT - mean_SSD

    return SSRT, mean_SSD

def getmaxRT(cond_df):
                
    #get TrialType column name
    trialType_col = 'TrialType'
    if 'TrialType' not in cond_df.columns: trialType_col='SS' #ADDED FOR MATZKE DATA
                
    #Set up keys for go trial types
    if 'GoCritical' in cond_df[trialType_col].unique(): #ADDED FOR TURK MOTOR SELECTIVE STOP
        goRT_key = 'GoCriticalRT'
    elif 'Target.RT' in cond_df.columns: #ADDED FOR MATZKE DATA
        goRT_key = 'Target.RT'
    else: 
        goRT_key = 'GoRT'
        
    maxRT = cond_df.loc[:, goRT_key].max()
    return(maxRT)

def calc_SSRT(subj_df, method='replacement', replacement_max=None):
    if method=='replacement':
        return calc_SSRT_wReplacement(subj_df, maxRT=replacement_max)
    elif method=='omission':
        return calc_SSRT_wOmission(subj_df)
    else:
        raise Exception("SSRT method must be either 'omission' or 'replacement'")
        
def filter_ssrt_subs(cond_df):

    # get SSD column name
    SSD_col = 'StopSignalDelay'
    if 'StopSignalDelay' not in cond_df.columns: SSD_col = 'SSD' #ADDED FOR MATZKE DATA
        
    #setting up stop failure RT key
    stopFailRT = 'StopFailureRT'
    if 'TargetDelay.RT' in cond_df.columns: stopFailRT = 'TargetDelay.RT' #ADDED FOR MATZKE DATA   

    keep_subs = []
    for subject in cond_df.Subject.unique():
        subdata = cond_df.query('Subject == %d' % subject).copy()
        #filter to stop trials with SSDs > 200
        longSSD_stops = subdata.loc[subdata[SSD_col] >= 200]
        #check for at least 1 stop success and 1 stop fail
        numStopFails = len(longSSD_stops.loc[longSSD_stops[stopFailRT]>0])
        numStopSuccs = len(longSSD_stops.loc[longSSD_stops[stopFailRT]==0])
        if (numStopFails>0) and (numStopSuccs>0):
            keep_subs.append(subject)
        else:
            continue
    return keep_subs
        
        
def get_ssrt_str(SSRTs, SSRTs_longSSDs, t, p):
    if p < 0.05:
        return f'''Overall SSRT (M = {np.mean(SSRTs):.0f} ms) was significantly slower than SSRT with short SSDs excluded (M = {np.mean(SSRTs_longSSDs):.0f} ms),
t({len(SSRTs_longSSDs)-1}) = {t:.02f}, {get_p_str(p)}'''
    else:
        if np.mean(SSRTs) > np.mean(SSRTs_longSSDs)+4:
            return f'''Overall SSRT (M = {np.mean(SSRTs):.0f} ms) was  numerically but not statistically significantly slower than SSRT with short SSDs excluded (M = {np.mean(SSRTs_longSSDs):.0f} ms),
t({len(SSRTs_longSSDs)-1}) = {t:.02f}, {get_p_str(p)}'''
        else:
            return f'''Overall SSRT (M = {np.mean(SSRTs):.0f} ms) was *NOT* slower than SSRT with short SSDs excluded (M = {np.mean(SSRTs_longSSDs):.0f} ms),
t({len(SSRTs_longSSDs)-1}) = {t:.02f}, {get_p_str(p)}'''


        
def get_p_str(p, thresh=0.001):
    if p < thresh:
        return 'p < 0.001'
    else:
        if f'{p:.2f}'=='0.00':
            return f'p = {p:.3f}'
        else:
            return f'p = {p:.2f}'
    
def ssrt_comparison(cond_dfs, conditions, method='replacement'):             
    
    for cond_df, condition  in zip(cond_dfs, conditions):
        replacement_maxRT = getmaxRT(cond_df)
        
        SSRTs = []
        SSRTs_longSSDs = []
        all_subjects = cond_df.Subject.unique()
        if len(all_subjects)==522: #if turk sample, swap out all_subjects
            all_subjects = full_passed_turkers
        usable_subjects = filter_ssrt_subs(cond_df.loc[cond_df.Subject.isin(all_subjects)]) #in case of turkers, only check those who passed
        print(formats.BOLD + condition + formats.END)
        print(f'{len(all_subjects) - len(usable_subjects)} subject(s) excluded from {condition}')
        for subject in usable_subjects:
            subdata = cond_df.query('Subject == %d' % subject).copy()
            #subset to ignore neg SSDs, keep NaNs( = go and ignore rows) - NO LONGER
            SSRT, _ = calc_SSRT(subdata, method=method, replacement_max=replacement_maxRT)
            SSRTs.append(SSRT)

            #Get SSRT after excluding SSDs < 200ms  
            subdata = subdata[(subdata.StopSignalDelay >= 200) | (subdata.StopSignalDelay.isnull())]
            SSRT, _ = calc_SSRT(subdata, method=method, replacement_max=replacement_maxRT)
            SSRTs_longSSDs.append(SSRT)


        t, p = stats.ttest_1samp(np.asarray(SSRTs)-np.asarray(SSRTs_longSSDs), 0)        
        print(get_ssrt_str(SSRTs, SSRTs_longSSDs, t, p))
        print(formats.BOLD + '*'*80 + formats.END)