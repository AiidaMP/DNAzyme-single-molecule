import pandas as pd
import numpy as np
import itertools as it

# ====== Loads GetEventsFunctions ======
# Make sure GetEventsFunctions is in the system path, or otherwise add it.
import sys

import_path = ""

if import_path not in sys.path:
    sys.path.append(import_path)

import GetEventsFunctions_withAA
import GetEventsFunctions
import GetEvents_ALEX

# ====== Even types preparation ======
# The following assume that traces have been discretized into 3 possible states.
# Declare the states for each of the different category of events.
#
# Events of type A and B come from the ALEX without AA channel, while C, D, E
# are new events
from GetEvents_ALEX import events_A, events_B

# Defining events in category C
donor = [(3, 1, 3), (3, 2, 3), (2, 1, 2), (2, 1, 3), (3, 1, 2)]
acceptor = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
alex = [(1, 2, 1), (1, 2, 2), (2, 2, 1)]

events_C = [(d, a, al) for d in donor for a in acceptor for al in alex]

# Defining events in category D
donor = [(3, 1, 1, 3), (3, 2, 2, 3), (2, 1, 1, 2)]
acceptor = [
    (1, 1, 1, 1),
    (2, 2, 2, 2),
    (3, 3, 3, 3),
    (1, 3, 1, 1),
    (1, 2, 1, 1),
    (1, 3, 1, 2),
    (2, 3, 1, 2),
    (2, 3, 1, 1),
    (2, 3, 2, 2),
    (1, 3, 2, 2),
]
alex = [(1, 2, 1, 1), (2, 2, 1, 1)]

events_D = [(d, a, al) for d in donor for a in acceptor for al in alex]

# Defining events in category E
donor = [(3, 1, 3), (3, 2, 3), (2, 1, 2), (2, 1, 3), (3, 1, 2)]
acceptor = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
alex = [(1, 1, 1), (2, 2, 2)]

events_E = [(d, a, al) for d in donor for a in acceptor for al in alex]


# ====== Noise robustness ======
# Events might not perfectly fit the states above, as they might be off by
# a few frames.
# Augment each to account for noise and off-by-x instances.
EVENTS_SIMPLE = events_C + events_D + events_E
events = [GetEventsFunctions_withAA.augment_event(d, a, x) for d, a, x in EVENTS_SIMPLE]


# ====== Indexize and group ======
# Each possible event is assigned a unique index, and further categorized into
# cleaving or non cleaving.
idx_a = list(range(0, len(events_A)))
idx_b = list(range(idx_a[-1] + 1, idx_a[-1] + 1 + len(events_B)))
idx_c = list(range(idx_b[-1] + 1, idx_b[-1] + 1 + len(events_C)))
idx_d = list(range(idx_c[-1] + 1, idx_c[-1] + 1 + len(events_D)))
idx_e = list(range(idx_d[-1] + 1, idx_d[-1] + 1 + len(events_E)))
non_clv_events = idx_a + idx_c + idx_e
clv_events = idx_b + idx_d
last_event_index = idx_e[-1] + 1


# ====== Helper functions ====== 
# Forwards the passed arguments to the GetEventsFunctions methods by adding
# the events specifically crafted above.
def get_unique_event_idx(separate_events):
    """
    Gets the last unique event index used across all calls.
    :param separate_events: All events processed so far.
    :return: The last valid used ID plus one.
    """
    global_eb_fret_counter = 1
    for df in separate_events:
        global_eb_fret_counter = max(global_eb_fret_counter, df["id"].max() + 1)
    return global_eb_fret_counter


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
    alex_fluo,
    donor_state,
    acceptor_state,
    alex_state,
    df_events_information,
):
    # Prepare temporal data for A/D test
    global_stats_2 = [
        {"donor": {y: [] for y in d}, "acceptor": {y: [] for y in a}}
        for d, a in GetEvents_ALEX.EVENTS_SIMPLE
    ]
    df_events_information_2 = pd.DataFrame(
        columns=[
            "Trace",
            "EventType",
            "EventStart",
            "EventEnd",
            "EventTotal",
            "TimesDonor",
            "TimesAcceptor",
            "FluoDonor",
            "FluoAcceptor",
        ]
    )

    df_separate_events_2 = [pd.DataFrame() for _ in GetEvents_ALEX.EVENTS_SIMPLE]
    df_ebfret_separate_events_2 = [
        pd.DataFrame(columns=["id", "donor", "acceptor", "alex"])
        for _ in GetEvents_ALEX.EVENTS_SIMPLE
    ]
    df_traces_events_2 = pd.DataFrame(columns=["Num"])
    df_events_2 = pd.DataFrame()

    # Call the non-AA version to get all the elements in categories A and B.
    GetEvents_ALEX.get_events_for(
        trace,
        None,
        global_stats_2,
        df_traces_events_2,
        df_events_2,
        df_ebfret_separate_events_2,
        df_separate_events_2,
        donor_fluo,
        acceptor_fluo,
        donor_state,
        acceptor_state,
        df_events_information_2,
        global_eb_fret_counter=get_unique_event_idx(df_ebfret_separate_events),
    )

    # Manually create traces event data
    df_traces_events.loc[trace - 1] = [f"Trace {trace}"] + [np.nan] * (
        df_traces_events.shape[1] - 1
    )

    # Loop through all the events to see if any has to be broken down into the
    # new C, D, E events.
    for _, event in df_events_information_2.iterrows():
        start = max(0, event["EventStart"] - 4)
        end = min(event["EventEnd"] + 4, len(donor_fluo))
        assert end > start, f"{start} after {end}"

        # A, B we keep.
        if (
            event["EventType"] in GetEvents_ALEX.idx_a
            or event["EventType"] in GetEvents_ALEX.idx_b
        ):
            # Write plot
            if event_plots is not None:
                event_plots[event["EventType"], event["EventEnd"]] = 1

            # Event info
            df_events_information.loc[df_events_information.shape[0]] = event

            # Trace info
            num_preexisting_events = np.sum(df_events_information["Trace"] == trace)
            while df_traces_events.shape[1] - 1 < num_preexisting_events:
                df_traces_events[df_traces_events.shape[1]] = np.nan

            df_traces_events.loc[trace - 1, num_preexisting_events] = (
                event["EventType"] + 1
            )

            # Events info
            this_donor_fluo = list(donor_fluo[event["EventStart"] : event["EventEnd"]])
            this_acceptor_fluo = list(
                acceptor_fluo[event["EventStart"] : event["EventEnd"]]
            )
            this_alex_fluo = list(alex_fluo[event["EventStart"] : event["EventEnd"]])
            this_time = min(
                event["EventEnd"] - event["EventStart"],
                len(this_donor_fluo),
                len(this_acceptor_fluo),
                len(this_alex_fluo),
            )

            # Create 3 new columns
            df_col = (df_events_information.shape[0] - 1) * 3
            df_events[df_col] = df_events[df_col + 1] = df_events[df_col + 2] = np.nan

            # Exopand rows
            nans = [np.nan] * df_events.shape[1]
            while df_events.shape[0] < this_time:
                df_events.loc[df_events.shape[0]] = nans

            # Save fluo values
            df_events.iloc[0:this_time, df_col] = this_donor_fluo[0:this_time]
            df_events.iloc[0:this_time, df_col + 1] = this_acceptor_fluo[0:this_time]
            df_events.iloc[0:this_time, df_col + 2] = this_alex_fluo[0:this_time]
        else:
            # Breaks down this event, no more and no less by using `t0` and `t1`,
            # into the new categories.
            GetEventsFunctions_withAA.get_events_for(
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
                alex_fluo,
                donor_state,
                acceptor_state,
                alex_state,
                df_events_information,
                start,
                end,
                idx_c[0],
                global_eb_fret_counter=get_unique_event_idx(
                    df_ebfret_separate_events_2
                ),
            )

    # Carry over events types A & B and add AA information
    def concat(merged, original, add_alex):
        if add_alex and not original.empty:
            original["alex"] = alex_fluo[original["time"]]
            original["alex_state"] = alex_state[original["time"]]

        return pd.concat((merged, original), axis=0, ignore_index=True)

    # Make sure A & B still are added to separate_events.
    for i in it.chain(idx_a, idx_b):
        df_ebfret_separate_events[i] = concat(
            df_ebfret_separate_events[i], df_ebfret_separate_events_2[i], True
        )
        df_separate_events[i] = concat(
            df_separate_events[i], df_separate_events_2[i], False
        )

    return (
        event_plots,
        global_stats,
        df_traces_events,
        df_events,
        df_ebfret_separate_events,
        df_separate_events,
        df_events_information,
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
    return GetEventsFunctions_withAA.get_dwell_times(
        idx_a, idx_b, idx_c, idx_d, idx_e, df_events_information
    )
