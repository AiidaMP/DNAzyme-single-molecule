import pandas as pd
import numpy as np

import sys

# ====== Loads GetEventsFunctions ====== 
# Make sure GetEventsFunctions is in the system path, or otherwise add it.
import_path = ""
if import_path not in sys.path:
    sys.path.append(import_path)
import GetEventsFunctions


# ====== Even types preparation ====== 
# The following assume that traces have been discretized into 3 possible states.
# Declare the states for each of the different category of events.

# Defining events in category A
donor = [(3, 1, 3), (3, 2, 3), (2, 1, 2)]
acceptor = [(1, 3, 1), (1, 2, 1), (2, 3, 2), (1, 3, 2), (2, 3, 1)]

events_A = [(d, a) for d in donor for a in acceptor]

# Defining events in category B
donor = [
    (3, 1, 2, 3),
    (3, 1, 1, 3),
    (3, 2, 2, 3),
    (2, 1, 1, 2),
    (2, 1, 3, 1),
    (2, 1, 1, 3),
    (3, 1, 1, 2),
]
acceptor = [
    (1, 2, 3, 1),
    (2, 2, 3, 2),
    (1, 1, 3, 1),
    (1, 1, 2, 1),
    (1, 1, 3, 2),
    (2, 2, 3, 1),
    (1, 2, 3, 2),
]

events_B = [(d, a) for d in donor for a in acceptor]

# Defining events in category C
donor = [(3, 1, 3), (3, 2, 3), (2, 1, 2)]
acceptor = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]

events_C = [(d, a) for d in donor for a in acceptor]

# Defining events in category D
donor = [(3, 1, 1, 3), (3, 2, 2, 3), (2, 1, 1, 2)]
acceptor = [
    (1, 3, 1, 1),
    (1, 2, 1, 1),
    (1, 3, 1, 2),
    (2, 3, 1, 2),
    (2, 3, 1, 1),
    (2, 3, 2, 2),
    (1, 3, 2, 2),
]

events_D = [(d, a) for d in donor for a in acceptor]

# Defining events in category E
donor = [(3, 2, 1, 1, 3)]
acceptor = [(1, 2, 2, 3, 1)]

events_E = [(d, a) for d in donor for a in acceptor]


# ====== Noise robustness ====== 
# Events might not perfectly fit the states above, as they might be off by
# a few frames.
# Augment each to account for noise and off-by-x instances.
EVENTS_SIMPLE = events_A + events_B + events_C + events_D
events = [GetEventsFunctions.augment_event(d, a) for d, a in EVENTS_SIMPLE]


# ====== Indexize and group ====== 
# Each possible event is assigned a unique index, and further categorized into
# cleaving or non cleaving.
idx_a = list(range(0, len(events_A)))
idx_b = list(range(idx_a[-1] + 1, idx_a[-1] + 1 + len(events_B)))
idx_c = list(range(idx_b[-1] + 1, idx_b[-1] + 1 + len(events_C)))
idx_d = list(range(idx_c[-1] + 1, idx_c[-1] + 1 + len(events_D)))
non_clv_events = idx_a + idx_c
clv_events = idx_b + idx_d


# ====== Helper functions ====== 
# Forwards the passed arguments to the GetEventsFunctions methods by adding
# the events specifically crafted above.
def get_events_for(
    trace,
    event_plots,
    global_stats,
    df_traces_events,
    df_events,
    df_ebfret_separate_events,
    df_separate_events,
    donor_fluo,
    acceptor_fluo,
    donor_state,
    acceptor_state,
    df_events_information,
    global_eb_fret_counter=None,
):
    return GetEventsFunctions.get_events_for(
        events,
        trace,
        event_plots,
        global_stats,
        df_traces_events,
        df_events,
        df_ebfret_separate_events,
        df_separate_events,
        donor_fluo,
        acceptor_fluo,
        donor_state,
        acceptor_state,
        df_events_information,
        global_eb_fret_counter=global_eb_fret_counter,
    )


def get_binary_cleavage(trace, df_events_information, extra_frames=4):
    return GetEventsFunctions.get_binary_cleavage(
        non_clv_events, trace, df_events_information, extra_frames
    )


def get_all_binary_cleavage(df_events_information, traces_data, extra_frames=4):
    return GetEventsFunctions.get_all_binary_cleavage(
        non_clv_events, df_events_information, traces_data, extra_frames
    )


def get_dwell_times(df_events_information):
    return GetEventsFunctions.get_dwell_times(
        idx_a, idx_b, idx_c, df_events_information
    )
