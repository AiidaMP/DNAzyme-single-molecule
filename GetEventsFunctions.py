import pandas as pd
import numpy as np


def augment_event(donor_e, acceptor_e):
    """
    Given changes in donor and acceptor that define on state, augments the
    valid state changes to account for possible noise or off-by-two frames.

    :param donor_e: State changes for donor
    :param acceptor_e: State changes for acceptor
    :return: Tuple with List of augmented changes for donor and acceptor
    """
    # Start with only the first state change.
    d_s = [[donor_e[0]]]
    a_s = [[acceptor_e[0]]]

    # Until we reach the event length.
    for i in range(1, len(donor_e)):
        new_as = []
        new_ds = []

        # For each of the already accounted for states, creates.
        for d, a in zip(d_s, a_s):
            # A even were the acceptor is repeated.
            new_as.append(a + [acceptor_e[i], acceptor_e[i]])
            new_ds.append(d + [d[-1], donor_e[i]])

            # A state were the acceptor is repeated twice.
            new_as.append(a + [acceptor_e[i], acceptor_e[i], acceptor_e[i]])
            new_ds.append(d + [d[-1], d[-1], donor_e[i]])

            # A state were the donor is repeated.
            new_as.append(a + [a[-1], acceptor_e[i]])
            new_ds.append(d + [donor_e[i], donor_e[i]])

            # A state were the donor is repeated twice.
            new_as.append(a + [a[-1], a[-1], acceptor_e[i]])
            new_ds.append(d + [donor_e[i], donor_e[i], donor_e[i]])

            # No state is repeated.
            new_as.append(a + [acceptor_e[i]])
            new_ds.append(d + [donor_e[i]])

        d_s = new_ds
        a_s = new_as

    return d_s, a_s


def get_times(state_changes, state_changes_time, event, idx, squash_consecutive=True):
    """
    Given an event and the times were it changed, get the accumulated time spent
    on each of the states individualy.

    :param state_changes: Discretized event states on changes
    :param state_changes_time: Time when it changed from one event to another
    :param event: The state changes previous too this event
    :param idx: Event unique index
    :param squash_consecutive: True if a state change to the same state gets
        counted as a single state.
    :return: The times spent on each of the states
    """
    times = []
    last_state = None

    # For each state change with respect to the previous even, count the time
    # spent there
    for st, se, d in zip(
        state_changes_time[idx:], state_changes_time[idx + 1 :], event
    ):
        total = se - st
        if squash_consecutive and last_state == d:
            times[-1] += total
        else:
            times.append(total)

        last_state = d

    # Likewise, for each current state change, count the stent time.
    for d, st, se in zip(
        state_changes[idx + len(event) :],
        state_changes_time[idx + len(event) :],
        state_changes_time[idx + len(event) + 1 :],
    ):
        if d != event[-1]:
            break

        total = se - st
        times[-1] += total

    return times


def squash_event(event):
    """
    Gets an event without two sequential states being the same.

    :param event: The event to squash
    :return: The squased event
    """
    e = [event[0]]
    for ei in event[1:]:
        if ei != e[-1]:
            e.append(ei)
    return e


def fix_states_2(trace, name, fluo, states, to_fix):
    """
    Given an event that has only 2 states, makes sure that y(state_0) < y(state_1).
    It can happen that states are not correctly represented in the data, but this
    script assumes that they are.

    :param trace: The whole trace to fix.
    :param name: Name of the trace, for logging only.
    :param fluo: Raw fluoresence data.
    :param states: The discretized fluoresence values.
    :param to_fix: The states that we are looking for and what to obtain.
    :return: The states respecting the invariants for state values.
    """
    # Obtain a rolling median of fluoresence value and the states that have to
    # be fixed.
    rolling_mean = fluo.rolling(1).median()
    a_idx_state_1 = states == to_fix[0]
    a_idx_state_2 = states == to_fix[1]

    # For each state, calculate the median also.
    a_median_state_1 = rolling_mean[a_idx_state_1].median()
    a_median_state_2 = rolling_mean[a_idx_state_2].median()

    # Now get the states so that they are ordered based on the raw values.
    state_idxs = sorted(
        zip([a_median_state_1, a_median_state_2], to_fix), key=lambda x: x[0]
    )
    state_idxs = list(zip(*state_idxs))[1]
    state_idxs = tuple([to_fix[state_idxs.index(i)] for i in to_fix])
    if tuple(state_idxs) != tuple(to_fix):
        print(
            f"TRACE/2 {trace} {name}, 1: {a_median_state_1:0.2f} / 2: {a_median_state_2:0.2f} -> {state_idxs}"
        )

    # Fix the states and return.
    states[a_idx_state_1] = state_idxs[0]
    states[a_idx_state_2] = state_idxs[1]
    return states


def fix_states_3(trace, name, fluo, states, to_fix):
    """
    Given an event that has only 3 states, makes sure that
        y(state_0) < y(state_1) < y(state_2).
    It can happen that states are not correctly represented in the data, but this
    script assumes that they are.

    See `fix_states_2` for a breakdown of the function itself.

    :param trace: The whole trace to fix.
    :param name: Name of the trace, for logging only.
    :param fluo: Raw fluoresence data.
    :param states: The discretized fluoresence values.
    :param to_fix: The states that we are looking for and what to obtain.
    :return: The states respecting the invariants for state values.
    """
    rolling_mean = fluo.rolling(1).median()
    a_idx_state_1 = states == to_fix[0]
    a_idx_state_2 = states == to_fix[1]
    a_idx_state_3 = states == to_fix[2]

    a_mean_state_1 = rolling_mean[a_idx_state_1].median()
    a_mean_state_2 = rolling_mean[a_idx_state_2].median()
    a_mean_state_3 = rolling_mean[a_idx_state_3].median()

    state_idxs = sorted(
        zip([a_mean_state_1, a_mean_state_2, a_mean_state_3], to_fix),
        key=lambda x: x[0],
    )
    state_idxs = list(zip(*state_idxs))[1]
    state_idxs = tuple([to_fix[state_idxs.index(i)] for i in to_fix])
    if tuple(state_idxs) != tuple(to_fix):
        print(
            f"TRACE/3 {trace} {name}, 1: {a_mean_state_1:0.2f} / 2: {a_mean_state_2:0.2f} / 3: {a_mean_state_3:0.2f} -> {state_idxs}"
        )

    states[a_idx_state_1] = state_idxs[0]
    states[a_idx_state_2] = state_idxs[1]
    states[a_idx_state_3] = state_idxs[2]
    return states


def fix_states(trace, name, fluo, states):
    """
    Given an event that has up to 4 states, makes sure that
        y(state_0) < y(state_1) < y(state_2) < y(state_3).
    It can happen that states are not correctly represented in the data, but this
    script assumes that they are.

    :param trace: The whole trace to fix.
    :param name: Name of the trace, for logging only.
    :param fluo: Raw fluoresence data.
    :param states: The discretized fluoresence values.
    :return: The states respecting the invariants for state values.
    """
    # Check how many unique states the trace has, and call either of the
    # specialized methods.
    unique_states = states.unique()
    state_1_found = 1 in unique_states
    state_2_found = 2 in unique_states
    state_3_found = 3 in unique_states
    state_4_found = 4 in unique_states

    if len(unique_states) == 1:
        return states

    if len(unique_states) == 2:
        if state_1_found and state_2_found:
            return fix_states_2(trace, name, fluo, states, (1, 2))

        if state_1_found and state_3_found:
            return fix_states_2(trace, name, fluo, states, (1, 3))

        if state_1_found and state_4_found:
            return fix_states_2(trace, name, fluo, states, (1, 4))

        if state_2_found and state_3_found:
            return fix_states_2(trace, name, fluo, states, (2, 3))

        if state_2_found and state_4_found:
            return fix_states_2(trace, name, fluo, states, (2, 4))

        if state_3_found and state_4_found:
            return fix_states_2(trace, name, fluo, states, (3, 4))

        # Dragons live here
        assert False, "WTF!"

    if len(unique_states) == 3:
        if state_1_found and state_2_found and state_3_found:
            return fix_states_3(trace, name, fluo, states, (1, 2, 3))

        if state_1_found and state_2_found and state_4_found:
            return fix_states_3(trace, name, fluo, states, (1, 2, 4))

        if state_1_found and state_3_found and state_4_found:
            return fix_states_3(trace, name, fluo, states, (1, 3, 4))

        if state_2_found and state_3_found and state_4_found:
            return fix_states_3(trace, name, fluo, states, (2, 3, 4))

        assert (
            False
        ), f"{state_1_found} and {state_2_found} and {state_3_found} and {state_4_found}"

    # See any of the `fix_states_x` for a breakdown of the code below.
    assert (
        len(unique_states) == 4
    ), f"{state_1_found} and {state_2_found} and {state_3_found} and {state_4_found}"

    rolling_mean = fluo.rolling(1).median()
    a_idx_state_1 = states == 1
    a_idx_state_2 = states == 2
    a_idx_state_3 = states == 3
    a_idx_state_4 = states == 4

    a_mean_state_1 = rolling_mean[a_idx_state_1].median()
    a_mean_state_2 = rolling_mean[a_idx_state_2].median()
    a_mean_state_3 = rolling_mean[a_idx_state_3].median()
    a_mean_state_4 = rolling_mean[a_idx_state_4].median()

    state_idxs = sorted(
        zip([a_mean_state_1, a_mean_state_2, a_mean_state_3, a_mean_state_4], range(4)),
        key=lambda x: x[0],
    )
    state_idxs = list(zip(*state_idxs))[1]
    state_idxs = tuple([state_idxs.index(i) + 1 for i in range(4)])
    if tuple(state_idxs) != (1, 2, 3, 4):
        print(
            f"TRACE/4 {trace} {name}, 1: {a_mean_state_1:0.2f} / 2: {a_mean_state_2:0.2f} / 3: {a_mean_state_3:0.2f} / 4: {a_mean_state_4:0.2f} -> {state_idxs}"
        )

    states[a_idx_state_1] = state_idxs[0]
    states[a_idx_state_2] = state_idxs[1]
    states[a_idx_state_3] = state_idxs[2]
    states[a_idx_state_4] = state_idxs[3]
    return states


def get_events_for(
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
    t0=None,
    t1=None,
    global_eb_fret_counter=None,
):
    """
    Given a trace and the list of all possible events to search for, construct
    dataframes with detailed information of what is found and where.

    :param events: The list of all possible events to search. Each event must be
        a tuple consisting of a unique index and the list of augmented events.
    :param trace: Unique identified of the trace.
    :param event_plots: Saves plottable data for each of the events found.
    :param global_stats: Can be used to save general information.
    :param df_traces_events: All the data for each event found.
    :param df_events: Acceptor and donor for the events found.
    :param df_ebfret_separate_events: Event indexes for a given trace.
    :param df_separate_events: For each event type, the occurrences found.
    :param donor_fluo: Donor values.
    :param acceptor_fluo: Acceptor values.
    :param df_events_information: Complete breakdown of the events found.
    :param t0: Time to start searching for events.
    :param t1: Time to stop searching for events.
    :param global_eb_fret_counter: Unique identifier seed value.
    """

    # Prepare data and structures needed.
    acceptor_state = fix_states(trace, "acceptor", acceptor_fluo, acceptor_state)
    donor_state = fix_states(trace, "donor", donor_fluo, donor_state)

    acceptor_state_changes = [None]
    donor_state_changes = [None]
    acceptor_state_changes_time = [None]
    donor_state_changes_time = [None]

    event_count = [0 for _ in events]
    ordered_events = []

    # Keep unique IDs for ebfret
    if global_eb_fret_counter is None:
        global_eb_fret_counter = 1
        for df in df_ebfret_separate_events:
            global_eb_fret_counter += df.shape[0]

    # Add to df
    df_traces_events.loc[trace - 1] = [f"Trace {trace}"] + [np.nan] * (
        df_traces_events.shape[1] - 1
    )

    # Loop through the whole trace.
    final_t = min(donor_state.shape[0], acceptor_state.shape[0])
    for t in range(1, final_t):
        if np.isnan(acceptor_state[t]):
            break

        # Check if there has been a state change that lasted for at least 2
        # frames.
        min_frames = 2
        acceptor_changes = (
            acceptor_state[t] != acceptor_state_changes[-1]
            and acceptor_state[min(final_t - 1, t + min_frames)] == acceptor_state[t]
        )
        donor_changes = (
            donor_state[t] != donor_state_changes[-1]
            and donor_state[min(final_t - 1, t + min_frames)] == donor_state[t]
        )
        changes = acceptor_changes or donor_changes

        if changes:
            # If there are any changes, see in which of Donor or Acceptor the
            # change occurred. Keep track of unique state values on each change.
            if acceptor_changes:
                acceptor_state_changes.append(acceptor_state[t])
                acceptor_state_changes_time.append(t)
            else:
                acceptor_state_changes.append(acceptor_state_changes[-1])
                acceptor_state_changes_time.append(t)

            if donor_changes:
                donor_state_changes.append(donor_state[t])
                donor_state_changes_time.append(t)
            else:
                donor_state_changes.append(donor_state_changes[-1])
                donor_state_changes_time.append(t)

            # Now that there has been a change, loop through all events and
            # seach if any matches.
            for idx, augmented_event in enumerate(events):
                found = False
                for donor, acceptor in zip(*augmented_event):
                    # Can't match if len does not match.
                    if len(donor) > len(donor_state_changes) - 1:
                        continue

                    # Can't match if len does not match.
                    if len(acceptor) > len(acceptor_state_changes) - 1:
                        continue

                    # Otherwise, match exact tuples.
                    if tuple(donor) != tuple(donor_state_changes[-len(donor) :]):
                        continue

                    if tuple(acceptor) != tuple(
                        acceptor_state_changes[-len(acceptor) :]
                    ):
                        continue

                    found = True
                    event_count[idx] += 1
                    break

                # Save all found events.
                if found:
                    start, end = (
                        donor_state_changes_time[-len(donor)],
                        donor_state_changes_time[-1],
                    )
                    assert end > start, f"{start} after {end}"
                    end += 4
                    total = end - start
                    ordered_events.append(
                        (
                            idx + 1,
                            start,
                            end,
                            total,
                            len(donor_state_changes_time) - len(donor),
                            donor,
                            acceptor,
                        )
                    )

    # Add last item N times to account for donor padding
    for i in range(15):
        donor_state_changes.append(donor_state_changes[-1])
        donor_state_changes_time.append(t)

        acceptor_state_changes.append(acceptor_state_changes[-1])
        acceptor_state_changes_time.append(t)

    # If there are overlapping events, keep the longest one within the overlap.
    previous_event = None
    filtered_events = []
    for current_event in ordered_events:
        eidx, estart, eend, etotal, idx, donor, acceptor = current_event

        if previous_event is not None:
            # Check the times each overlapping events lasted.
            times_donor = get_times(
                donor_state_changes, donor_state_changes_time, previous_event[-2], idx
            )
            times_acceptor = get_times(
                acceptor_state_changes,
                acceptor_state_changes_time,
                previous_event[-1],
                idx,
            )

            # Take some initial time
            initial_time = 0
            if len(times_donor) > 1:
                initial_time = times_donor[0]
            elif len(times_acceptor) > 1:
                initial_time = times_acceptor[0]

            # Check if the ovearlap happened for these two events.
            if not (estart + initial_time <= previous_event[2] <= eend):
                filtered_events.append(previous_event)

                while df_traces_events.shape[1] - 1 < len(filtered_events):
                    df_traces_events[df_traces_events.shape[1]] = np.nan

                df_traces_events.loc[trace - 1, len(filtered_events)] = previous_event[
                    0
                ]

                if event_plots is not None:
                    event_plots[previous_event[0] - 1, previous_event[2]] = 1

        previous_event = current_event

    # Always add last, overwrite ordered with filtered
    if previous_event is not None:
        filtered_events.append(previous_event)
        ordered_events = filtered_events

        while df_traces_events.shape[1] - 1 < len(filtered_events):
            df_traces_events[df_traces_events.shape[1]] = np.nan

        df_traces_events.loc[trace - 1, len(filtered_events)] = previous_event[0]

        if event_plots is not None:
            event_plots[previous_event[0] - 1, previous_event[2]] = 1

    # Finally, for each event, fill all the dataframes information
    df_col = df_events.shape[1]
    events_information = []
    squash_consecutive = False
    for eidx, estart, eend, etotal, idx, donor, acceptor in ordered_events:
        if squash_consecutive:
            times_donor = get_times(
                donor_state_changes,
                donor_state_changes_time,
                donor,
                idx,
                squash_consecutive,
            )
            assert len(times_donor) == len(
                squash_event(donor)
            ), "Donor time not matching"

            times_acceptor = get_times(
                acceptor_state_changes,
                acceptor_state_changes_time,
                acceptor,
                idx,
                squash_consecutive,
            )
            assert len(times_acceptor) == len(
                squash_event(acceptor)
            ), f"Acceptor time not matching {times_acceptor} != {squash_event(acceptor)}"
        else:
            times_donor = get_times(
                donor_state_changes,
                donor_state_changes_time,
                donor,
                idx,
                squash_consecutive,
            )
            assert len(times_donor) == len(donor), "Donor time not matching"

            times_acceptor = get_times(
                acceptor_state_changes,
                acceptor_state_changes_time,
                acceptor,
                idx,
                squash_consecutive,
            )
            assert len(times_acceptor) == len(
                acceptor
            ), f"Acceptor time not matching {times_acceptor} != {acceptor}"

        events_information.append(
            (eidx, estart, eend, etotal, times_donor, times_acceptor)
        )

        # Create columns
        df_events[df_col] = df_events[df_col + 1] = np.nan

        # Get fluoresence
        this_donor_fluo = list(donor_fluo[estart:eend])
        this_acceptor_fluo = list(acceptor_fluo[estart:eend])
        this_time = min(eend - estart, len(this_donor_fluo), len(this_acceptor_fluo))

        this_donor_fluo = list(donor_fluo[estart : estart + this_time])
        this_acceptor_fluo = list(acceptor_fluo[estart : estart + this_time])

        # Add detailed information
        assert eend > estart, f"{estart} after {eend}"
        df_events_information.loc[df_events_information.shape[0]] = [
            trace,
            eidx - 1,
            estart,
            eend,
            etotal,
            (times_donor, donor),
            (times_acceptor, acceptor),
            this_donor_fluo,
            this_acceptor_fluo,
        ]

        # Extend rows
        nans = [np.nan] * df_events.shape[1]
        while df_events.shape[0] < this_time:
            df_events = df_events.append(nans)

        # Save fluo values
        df_events.iloc[0:this_time, df_col] = this_donor_fluo
        df_events.iloc[0:this_time, df_col + 1] = this_acceptor_fluo
        df_col += 2

        # PER EVENT
        df_this_event = df_separate_events[eidx - 1]
        df_event_col = df_this_event.shape[1]

        # Create columns
        df_this_event[df_event_col] = df_this_event[df_event_col + 1] = np.nan

        # Extend rows
        nans = [np.nan] * df_this_event.shape[1]
        while df_this_event.shape[0] < this_time:
            df_separate_events[eidx - 1] = df_this_event.append(nans)
            df_this_event = df_separate_events[eidx - 1]

        # Save fluo values
        df_this_event.iloc[0:this_time, df_event_col] = this_donor_fluo
        df_this_event.iloc[0:this_time, df_event_col + 1] = this_acceptor_fluo

        # PER EVENT, EbFret like
        ebfret_like_df = pd.DataFrame(
            {
                "id": float(global_eb_fret_counter),
                "donor": donor_fluo[estart:eend],
                "acceptor": acceptor_fluo[estart:eend],
                "donor_state": donor_state[estart:eend],
                "acceptor_state": acceptor_state[estart:eend],
                "time": list(range(estart, estart + len(acceptor_state[estart:eend]))),
            }
        )
        df_ebfret_separate_events[eidx - 1] = df_ebfret_separate_events[
            eidx - 1
        ].append(ebfret_like_df)
        global_eb_fret_counter += 1

    # Can't match if len does not match.
    return (
        event_plots,
        global_stats,
        df_traces_events,
        df_events,
        df_ebfret_separate_events,
        df_separate_events,
        df_events_information,
    )


def get_binary_cleavage(non_clv_events, trace, df_events_information, extra_frames=4):
    """
    For a given trace where all events are found, fills a binary sequence where
    1 means a cleavage happened.

    :param non_clv_events: Events that are not considered cleaving.
    :param trace: Unique trace identifier.
    :param df_events_information: Detailed information of the events found.
    :param extra_frames: Padding to account for noise and off-by-x errors.
    :return: Binary sequence for cleavage vs non cleavage.
    """
    # Only consider one trace.
    trace_data = df_events_information[df_events_information["Trace"] == trace]
    trace_data = trace_data.reset_index()

    # Break each event down to clv vs non-clv already.
    event_binary = 0 if trace_data.loc[0, "EventType"] in non_clv_events else 1
    event_end = trace_data.loc[0, "EventEnd"]
    event_start = trace_data.loc[0, "EventStart"]

    # Pad with 0s until the first cleaveage happens.
    x = [0] * event_start
    x += [event_binary] * (event_end - event_start - extra_frames)

    # Loop through the rest of events.
    for i in range(1, trace_data.shape[0]):
        event_type = trace_data.loc[i, "EventType"]
        event_end = trace_data.loc[i, "EventEnd"]
        event_start = trace_data.loc[i, "EventStart"]
        event_binary = 0 if event_type in non_clv_events else 1

        times_donor, _ = trace_data.loc[i, "TimesDonor"]
        times_acceptor, _ = trace_data.loc[i, "TimesAcceptor"]
        first_change_duration = 0
        if len(times_donor) > 1:
            first_change_duration = times_donor[0]
        elif len(times_acceptor) > 1:
            first_change_duration = times_acceptor[0]

        # Fill space between events
        x += [0] * max(
            0, event_start - (trace_data.loc[i - 1, "EventEnd"] - extra_frames)
        )

        # Fill 0s the first "step"
        x += [0] * first_change_duration

        # Fill the rest with the event binary type
        x += [event_binary] * (
            (event_end - extra_frames) - event_start - first_change_duration
        )

    return x


def get_all_binary_cleavage(
    non_clv_events, df_events_information, traces_data, extra_frames=4
):
    """
    For all traces, construct binary sequences with cleavage information.

    :param non_clv_events: Events that are considered not to cause cleavage.
    :param df_events_information: All found events informations.
    :param traces_data: Data for all traces.
    :param extra_frames: Frames to account for noise and off-by-x errors.
    """
    # Get cleavage sequence for each trace.
    df_all_data = pd.DataFrame()
    for trace in df_events_information["Trace"].unique():
        clv = get_binary_cleavage(
            non_clv_events, trace, df_events_information, extra_frames
        )

        # Fill with 0s to trace length
        trace_length = traces_data[traces_data["Trace"] == trace].shape[0]
        clv += [0] * (trace_length - len(clv))

        # Create columns
        while df_all_data.shape[1] < len(clv):
            df_all_data[df_all_data.shape[1]] = np.nan

        # Fill in dataframe
        clv += [np.nan] * (df_all_data.shape[1] - len(clv))
        df_all_data.loc[trace] = clv

    return df_all_data


def get_dwell_times(idxs_a, idxs_b, idxs_c, df_events_information):
    """
    For each state of an event type, it gets the duration of that event.

    :param idxs_a: Unique ids for events of type A.
    :param idxs_a: Unique ids for events of type B.
    :param idxs_a: Unique ids for events of type C.
    :param df_events_information: All found events' information.
    :return: Tuple with all data found and the broken down information for each
        trace.
    """
    As = []
    B1s = []
    B2s = []
    Cs = []

    As_t = []
    Bs_t = []
    Cs_t = []

    # Loop through all the events found.
    for _, row in df_events_information.iterrows():
        trace = row["Trace"]
        times_acceptor, acceptor = row["TimesAcceptor"]
        times_donor, donor = row["TimesDonor"]

        if row["EventType"] in idxs_a:
            # Seek acceptor change first
            acceptor_pos = 0
            initial_acceptor = acceptor[0]
            initial_acceptor = acceptor[acceptor_pos]
            while acceptor[acceptor_pos] == initial_acceptor:
                acceptor_pos += 1

            time = 0
            initial_pos = acceptor_pos
            while acceptor[acceptor_pos] == acceptor[initial_pos]:
                time += times_acceptor[acceptor_pos]
                acceptor_pos += 1

            As.append(time)
            As_t.append(trace)

        elif row["EventType"] in idxs_c:
            # Seek donor change first
            donor_pos = 0
            initial_donor = donor[0]
            initial_donor = donor[donor_pos]
            while donor[donor_pos] == initial_donor:
                donor_pos += 1

            time = 0
            initial_pos = donor_pos
            while donor[donor_pos] == donor[initial_pos]:
                time += times_donor[donor_pos]
                donor_pos += 1

            Cs.append(time)
            Cs_t.append(trace)


        elif row["EventType"] in idxs_b:

            # Seek donor change first
            donor_pos = 0
            initial_donor = donor[donor_pos]
            while donor[donor_pos] == initial_donor:
                donor_pos += 1

            # Are we already in the acceptor b1 or before it?
            num_changes = 0
            for a, b in zip(acceptor[donor_pos:], acceptor[donor_pos + 1 :]):
                num_changes += int(a != b)

            # If not, add current and move forward
            b1 = 0
            if num_changes >= 3:
                b1 += times_acceptor[donor_pos]
                donor_pos += 1

            # Add B1 as long as acceptor doesn't change
            initial_pos = donor_pos
            acceptor_pos = donor_pos
            while acceptor[acceptor_pos] == acceptor[initial_pos]:
                b1 += times_acceptor[acceptor_pos]
                acceptor_pos += 1

            # Add B2 as long as acceptor doesn't change
            initial_pos = acceptor_pos
            b2 = 0
            while acceptor[acceptor_pos] == acceptor[initial_pos]:
                b2 += times_acceptor[acceptor_pos]
                acceptor_pos += 1

            B1s.append(b1)
            B2s.append(b2)
            Bs_t.append(trace)

    # Put everything in a DataFrame by taking the longest sequence as the number
    # of rows on the df.
    maxlen = max(len(As), len(B1s), len(B2s), len(Cs))
    As = As + [np.nan] * (maxlen - len(As))
    B1s = B1s + [np.nan] * (maxlen - len(B1s))
    B2s = B2s + [np.nan] * (maxlen - len(B2s))
    Cs = Cs + [np.nan] * (maxlen - len(Cs))

    As_t = As_t + [np.nan] * (maxlen - len(As_t))
    Bs_t = Bs_t + [np.nan] * (maxlen - len(Bs_t))
    Cs_t = Cs_t + [np.nan] * (maxlen - len(Cs_t))

    df_data = pd.DataFrame({"A": As, "B1": B1s, "B2": B2s, "C": Cs})
    df_traces = pd.DataFrame(
        {"At": As_t, "A": As, "Bt": Bs_t, "B1": B1s, "B2": B2s, "Ct": Cs_t, "C": Cs}
    )

    return df_data, df_traces
