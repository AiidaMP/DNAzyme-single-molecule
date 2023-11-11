import pandas as pd
import numpy as np
from GetEvents_ALEX_wAA import get_dwell_times
import GetEvents_ALEX_wAA
import warnings
warnings.filterwarnings('ignore')


# Import data
videos = [
    ]

for video in videos:
    print(video)
    # Opening the files containing the fluorescence states for the three different signals (DD-Donor, DA-Acceptor, AA-ALEX)
    Donor_Data  = pd.read_csv(video.format('TracesDD.csv'))
    Acceptor_Data  = pd.read_csv(video.format('TracesDA.csv'))
    ALEX_Data  = pd.read_csv(video.format('TracesAA.csv'))
    
    
    row, col = np.shape(Donor_Data)
    traces = len(Donor_Data['Trace'].unique())
      
    
    global_stats = [
        {
          'donor': {y: [] for y in d},
          'acceptor': {y: [] for y in a},
          'alex': {y: [] for y in x}
        }
        for d, a, x in GetEvents_ALEX_wAA.EVENTS_SIMPLE
    ]
    
    df_events_information = pd.DataFrame(columns=[
        'Trace', 'EventType', 'EventStart', 'EventEnd', 'EventTotal', 'TimesDonor', 'TimesAcceptor', 'TimesAA', 'FluoDonor', 'FluoAcceptor', 'FluoAA'
    ])
            
    df_separate_events = [pd.DataFrame() for _ in range(GetEvents_ALEX_wAA.last_event_index)]
    df_ebfret_separate_events = [pd.DataFrame(columns=['id', 'donor', 'acceptor', 'alex']) for _ in range(GetEvents_ALEX_wAA.last_event_index)]
    
    df_traces_events = pd.DataFrame(columns=['Num'])
    df_traces_ev_time = pd.DataFrame(columns=['Num'])
    df_events = pd.DataFrame()
    
    
    for trace in range(1, traces + 1):

        if (Donor_Data['Trace'] == trace).any() and not (Acceptor_Data['Trace'] == trace).any():
            print(f'******** TRACE {trace} is missing Acceptor data')
            continue
        
        if not (Donor_Data['Trace'] == trace).any() and (Acceptor_Data['Trace'] == trace).any():
            print(f'******** TRACE {trace} is missing Donor data')
            continue
        
        if not (Donor_Data['Trace'] == trace).any() and ((Acceptor_Data['Trace'] == trace).any() or (ALEX_Data['Trace'] == trace).any()):
            print(f'******** TRACE {trace} is missing ALEX data')
            continue
        
        trace_donor_data = Donor_Data[Donor_Data['Trace'] == trace].copy()
        if trace_donor_data.shape[0] == 0:
            continue
        
        trace_donor_data.index = range(trace_donor_data.shape[0])
        donor_fluo = trace_donor_data['Donor']
        donor_state = trace_donor_data['State']
        
        trace_acceptor_data = Acceptor_Data[Acceptor_Data['Trace'] == trace].copy()
        trace_acceptor_data.index = range(trace_acceptor_data.shape[0])
        acceptor_fluo = trace_acceptor_data['Acceptor']
        acceptor_state = trace_acceptor_data['State']
        
        trace_alex_data = ALEX_Data[ALEX_Data['Trace'] == trace].copy()
        trace_alex_data.index = range(trace_alex_data.shape[0])
        alex_fluo = trace_alex_data['Alex']
        alex_state = trace_alex_data['State']
        
        event_plots = None
        
        # PHASE 1 (donor, acceptor)
        GetEvents_ALEX_wAA.get_events_for(trace, 
            event_plots, global_stats, df_traces_events, df_events, df_ebfret_separate_events, df_separate_events, 
            donor_fluo, acceptor_fluo, alex_fluo,
            donor_state, acceptor_state, alex_state, 
            df_events_information)
    
    # Data frame describing the traces binary: frames corresponding to cleavage events represented by 1, while all
    # the other represented by 0    
    df_binary_cleavage = GetEvents_ALEX_wAA.get_all_binary_cleavage(df_events_information, Donor_Data)

    df_binary_cleavage.to_csv(video.format('BinaryCleavage.csv'), header = False, sep=",")
    df_binary_cleavage.to_excel(video.format('BinaryCleavage.xlsx'), header = False)
    
    # Data frame containing the dwell time of each event separated by categories and a second data frame in
    # which the trace number from that event is kept for traceability        
    df_dwell_times, df_dwell_times_per_trace = get_dwell_times(df_events_information)
   
    df_dwell_times.to_csv(video.format('DwellTimes.csv'), header = False, index = None, sep=",")
    df_dwell_times.to_excel(video.format('DwellTimes.xlsx'), header = False, index = None)
    df_dwell_times_per_trace.to_csv(video.format('DwellTimesPerTrace.csv'), sep=",", index = None)
    df_dwell_times_per_trace.to_excel(video.format('DwellTimesPerTrace.xlsx'), index = None)
    
    
    # Grouping the different events into the corresponding categories
    final_A = len(GetEvents_ALEX_wAA.events_A)
    final_B = final_A + len(GetEvents_ALEX_wAA.events_B)
    final_C = final_B + len(GetEvents_ALEX_wAA.events_C)
    final_D = final_C + len(GetEvents_ALEX_wAA.events_D)
    final_E = final_D + len(GetEvents_ALEX_wAA.events_E)
    
    
    max_e_per_trace = len(df_traces_events.columns)-1
    
    cat_group = df_traces_events.copy()
    for i in range(1,max_e_per_trace + 1,1):
        i = str(i)
        start = 1
        end = final_A
        cat_group.loc[(start <= df_traces_events[i]) & (df_traces_events[i] <= end), i] = 'A'
        end += final_B
        start += final_A
        cat_group.loc[(start <= df_traces_events[i]) & (df_traces_events[i] <= end), i] = 'B'
        end += final_C
        start += final_B
        cat_group.loc[(start <= df_traces_events[i]) & (df_traces_events[i] <= end), i] = 'C'
        end += final_D
        start += final_C
        cat_group.loc[(start <= df_traces_events[i]) & (df_traces_events[i] <= end), i] = 'D'
        end += final_E
        start += final_D
        cat_group.loc[(start <= df_traces_events[i]) & (df_traces_events[i] <= end), i] = 'E'
    
    # Exporting the categorization of the events for each of the traces
    cat_group.to_csv(video.format('Categories_Events.csv'), index=None, sep=",")
    cat_group.to_excel(video.format('Categories_Events.xlsx'), index=None)
       
    print('CATEGORIES EVENTS SAVED')
    
    
    # Exporting the fluorescent traces for each category separately
    df_ebfret_events_A = pd.concat([x.reset_index(drop=True) for x in df_ebfret_separate_events[0:final_A]], axis = 0)
    df_ebfret_events_B = pd.concat([x.reset_index(drop=True) for x in df_ebfret_separate_events[final_A:final_B]], axis = 0)
    df_ebfret_events_C = pd.concat([x.reset_index(drop=True) for x in df_ebfret_separate_events[final_B:final_C]], axis = 0)
    df_ebfret_events_D = pd.concat([x.reset_index(drop=True) for x in df_ebfret_separate_events[final_C:final_D]], axis = 0)
    df_ebfret_events_E = pd.concat([x.reset_index(drop=True) for x in df_ebfret_separate_events[final_D:final_E]], axis = 0)
    
    df_ebfret_events_A.to_csv('SeparateEvents/' + video.format('Fluo_Only_Event_A.csv'), index=None, sep=",")  
    df_ebfret_events_B.to_csv('SeparateEvents/' + video.format('Fluo_Only_Event_B.csv'), index=None, sep=",")    
    df_ebfret_events_C.to_csv('SeparateEvents/' + video.format('Fluo_Only_Event_C.csv'), index=None, sep=",")    
    df_ebfret_events_D.to_csv('SeparateEvents/' + video.format('Fluo_Only_Event_D.csv'), index=None, sep=",")    
    df_ebfret_events_E.to_csv('SeparateEvents/' + video.format('Fluo_Only_Event_E.csv'), index=None, sep=",")
