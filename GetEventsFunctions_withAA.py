import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from GetEventsFunctions import get_times, squash_event, fix_states


def augment_event(donor_e, acceptor_e, alexAA_e):
    """
    Given changes in donor and acceptor that define on state, augments the
    valid state changes to account for possible noise or off-by-two frames.

    :param donor_e: State changes for donor
    :param acceptor_e: State changes for acceptor
    :return: Tuple with List of augmented changes for donor and acceptor
    """
    d_s = [[donor_e[0]]]
    a_s = [[acceptor_e[0]]]
    x_s = [[alexAA_e[0]]]

    # Until we reach the event length.
    for i in range(1, len(donor_e)):
        new_as = []
        new_ds = []
        new_xs = []

        # For each of the already accounted for states, creates.
        for d, a, x in zip(d_s, a_s, x_s):
            # Repeat acceptor
            new_as.append(a + [acceptor_e[i], acceptor_e[i]])
            new_ds.append(d + [d[-1], donor_e[i]])
            new_xs.append(x + [x[-1], alexAA_e[i]])

            # Repeat acceptor x2
            new_as.append(a + [acceptor_e[i], acceptor_e[i], acceptor_e[i]])
            new_ds.append(d + [d[-1], d[-1], donor_e[i]])
            new_xs.append(x + [x[-1], x[-1], alexAA_e[i]])

            # Repeat donor
            new_as.append(a + [a[-1], acceptor_e[i]])
            new_ds.append(d + [donor_e[i], donor_e[i]])
            new_xs.append(x + [x[-1], alexAA_e[i]])

            # Repeat donor x2
            new_as.append(a + [a[-1], a[-1], acceptor_e[i]])
            new_ds.append(d + [donor_e[i], donor_e[i], donor_e[i]])
            new_xs.append(x + [x[-1], x[-1], alexAA_e[i]])

            # Repeat Ax
            new_as.append(a + [a[-1], acceptor_e[i]])
            new_ds.append(d + [d[-1], donor_e[i]])
            new_xs.append(x + [alexAA_e[i], alexAA_e[i]])

            # Repeat Ax x2
            new_as.append(a + [a[-1], a[-1], acceptor_e[i]])
            new_ds.append(d + [d[-1], d[-1], donor_e[i]])
            new_xs.append(x + [alexAA_e[i], alexAA_e[i], alexAA_e[i]])

            # Same
            new_as.append(a + [acceptor_e[i]])
            new_ds.append(d + [donor_e[i]])
            new_xs.append(x + [alexAA_e[i]])

        d_s = new_ds
        a_s = new_as
        x_s = new_xs

    d_s = tuple([tuple(x) for x in d_s])
    a_s = tuple([tuple(x) for x in a_s])
    x_s = tuple([tuple(x) for x in x_s])
    return d_s, a_s, x_s


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
    alex_fluo,
    donor_state,
    acceptor_state,
    alex_state,
    df_events_information,
    t0,
    t1,
    id_shift,
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
    :param alex_fluo: Alex values.
    :param donor_state: State for donor carried from non-AA.
    :param acceptor_state: State for acceptor carried from non-AA.
    :param alex_state: State for alex carried from non-AA.
    :param df_events_information: Complete breakdown of the events found.
    :param t0: Time to start searching for events.
    :param t1: Time to stop searching for events.
    :param id_shift: Given that IDs from non-AA and AA are not continuous.
    :param global_eb_fret_counter: Unique identifier seed value.
    """

    # Prepare data and structures needed.
    acceptor_state = fix_states(trace, "acceptor", acceptor_fluo, acceptor_state)
    donor_state = fix_states(trace, "donor", donor_fluo, donor_state)
    alex_state = fix_states(trace, "alex", alex_fluo, alex_state)

    acceptor_state_changes = [None]
    donor_state_changes = [None]
    alex_state_changes = [None]
    acceptor_state_changes_time = [None]
    donor_state_changes_time = [None]
    alex_state_changes_time = [None]

    num_preexisting_events = np.sum(df_events_information["Trace"] == trace)
    event_count = [0 for _ in events]
    ordered_events = []

    # Keep unique IDs for ebfret
    if global_eb_fret_counter is None:
        global_eb_fret_counter = 1
        for df in df_ebfret_separate_events:
            global_eb_fret_counter += df.shape[0]

    # Add to df
    if trace - 1 not in df_traces_events.index:
        df_traces_events.loc[trace - 1] = [f"Trace {trace}"] + [np.nan] * (
            df_traces_events.shape[1] - 1
        )

    # Loop through the whole trace or the specified range.
    final_t = min(donor_state.shape[0], acceptor_state.shape[0], alex_state.shape[0])
    for t in range(t0 or 1, min(t1 or final_t, final_t)):
        if np.isnan(acceptor_state[t]):
            break

        # Check if there has been a state change that lasted for at least 2
        # frames.
        min_frames = 2
        acceptor_has_changes = (
            acceptor_state[t] != acceptor_state_changes[-1]
            and acceptor_state[min(final_t - 1, t + min_frames)] == acceptor_state[t]
        )
        donor_has_changes = (
            donor_state[t] != donor_state_changes[-1]
            and donor_state[min(final_t - 1, t + min_frames)] == donor_state[t]
        )
        alex_has_changes = (
            alex_state[t] != alex_state_changes[-1]
            and alex_state[min(final_t - 1, t + min_frames)] == alex_state[t]
        )
        changes = acceptor_has_changes or donor_has_changes or alex_has_changes

        if changes:
            # If there are any changes, see in which of Donor or Acceptor the
            # change occurred. Keep track of unique state values on each change.
            if acceptor_has_changes:
                acceptor_state_changes.append(acceptor_state[t])
                acceptor_state_changes_time.append(t)
            else:
                acceptor_state_changes.append(acceptor_state_changes[-1])
                acceptor_state_changes_time.append(t)

            if donor_has_changes:
                donor_state_changes.append(donor_state[t])
                donor_state_changes_time.append(t)
            else:
                donor_state_changes.append(donor_state_changes[-1])
                donor_state_changes_time.append(t)

            if alex_has_changes:
                alex_state_changes.append(alex_state[t])
                alex_state_changes_time.append(t)
            else:
                alex_state_changes.append(alex_state_changes[-1])
                alex_state_changes_time.append(t)

            # Now that there has been a change, loop through all events and
            # seach if any matches.
            for idx, augmented_event in enumerate(events):
                found = False
                for donor, acceptor, alex in zip(*augmented_event):
                    # First check for length matches.
                    if len(donor) > len(donor_state_changes) - 1:
                        continue

                    if len(acceptor) > len(acceptor_state_changes) - 1:
                        continue

                    if len(alex) > len(alex_state_changes) - 1:
                        continue

                    # Then for the tuples themselves.
                    if donor != tuple(donor_state_changes[-len(donor) :]):
                        continue

                    if acceptor != tuple(acceptor_state_changes[-len(acceptor) :]):
                        continue

                    if alex != tuple(alex_state_changes[-len(alex) :]):
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
                    end += 4
                    total = end - start
                    ordered_events.append(
                        (
                            idx + 1 + id_shift,
                            start,
                            end,
                            total,
                            len(donor_state_changes_time) - len(donor),
                            donor,
                            acceptor,
                            alex,
                        )
                    )

    # Add last item N times to account for donor padding
    for i in range(15):
        donor_state_changes.append(donor_state_changes[-1])
        donor_state_changes_time.append(t)

        acceptor_state_changes.append(acceptor_state_changes[-1])
        acceptor_state_changes_time.append(t)

        alex_state_changes.append(alex_state_changes[-1])
        alex_state_changes_time.append(t)

    # If there are overlaps, keep the longest one
    previous_event = None
    filtered_events = []
    for current_event in ordered_events:
        eidx, estart, eend, etotal, idx, donor, acceptor, alex = current_event

        if previous_event is not None:
            # Check the times each overlapping events lasted.
            times_donor = get_times(
                donor_state_changes, donor_state_changes_time, previous_event[-3], idx
            )
            times_acceptor = get_times(
                acceptor_state_changes,
                acceptor_state_changes_time,
                previous_event[-2],
                idx,
            )
            times_alex = get_times(
                alex_state_changes, alex_state_changes_time, previous_event[-1], idx
            )

            # Take some initial time
            initial_time = 0
            if len(times_donor) > 1:
                initial_time = times_donor[0]
            elif len(times_acceptor) > 1:
                initial_time = times_acceptor[0]
            elif len(times_alex) > 1:
                initial_time = times_alex[0]

            # Check if the ovearlap happened for these two events.
            if not (estart + initial_time <= previous_event[2] <= eend):
                filtered_events.append(previous_event)

                while (
                    df_traces_events.shape[1] - 1
                    < len(filtered_events) + num_preexisting_events
                ):
                    df_traces_events[df_traces_events.shape[1]] = np.nan

                df_traces_events.loc[
                    trace - 1, len(filtered_events) + num_preexisting_events
                ] = previous_event[0]

                if event_plots is not None:
                    event_plots[previous_event[0] - 1, previous_event[2]] = 1

        previous_event = current_event

    # Always add last, overwrite ordered with filtered
    if previous_event is not None:
        filtered_events.append(previous_event)
        ordered_events = filtered_events

        while (
            df_traces_events.shape[1] - 1
            < len(filtered_events) + num_preexisting_events
        ):
            df_traces_events[df_traces_events.shape[1]] = np.nan

        df_traces_events.loc[
            trace - 1, len(filtered_events) + num_preexisting_events
        ] = previous_event[0]

        if event_plots is not None:
            event_plots[previous_event[0] - 1, previous_event[2]] = 1

    # Finally, for each event, fill all the dataframes information
    df_col = df_events.shape[1]
    events_information = []
    squash_consecutive = False
    for eidx, estart, eend, etotal, idx, donor, acceptor, alex in ordered_events:
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

            times_alex = get_times(
                alex_state_changes,
                alex_state_changes_time,
                alex,
                idx,
                squash_consecutive,
            )
            assert len(times_alex) == len(
                squash_event(alex)
            ), f"ALEX time not matching {times_alex} != {squash_event(alex)}"
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

            times_alex = get_times(
                alex_state_changes,
                alex_state_changes_time,
                alex,
                idx,
                squash_consecutive,
            )
            assert len(times_alex) == len(
                alex
            ), f"ALEX time not matching {times_alex} != {alex}"

        events_information.append(
            (eidx, estart, eend, etotal, times_donor, times_acceptor, times_alex)
        )

        # Create columns
        df_events[df_col] = df_events[df_col + 1] = df_events[df_col + 2] = np.nan

        # Get fluoresence
        this_donor_fluo = list(donor_fluo[estart:eend])
        this_acceptor_fluo = list(acceptor_fluo[estart:eend])
        this_alex_fluo = list(alex_fluo[estart:eend])
        this_time = min(
            eend - estart,
            len(this_donor_fluo),
            len(this_acceptor_fluo),
            len(this_alex_fluo),
        )

        this_donor_fluo = list(donor_fluo[estart : estart + this_time])
        this_acceptor_fluo = list(acceptor_fluo[estart : estart + this_time])
        this_alex_fluo = list(alex_fluo[estart : estart + this_time])

        # Add detailed information
        df_events_information.loc[df_events_information.shape[0]] = [
            trace,
            eidx - 1,
            estart,
            eend,
            etotal,
            (times_donor, donor),
            (times_acceptor, acceptor),
            (times_alex, alex),
            this_donor_fluo,
            this_acceptor_fluo,
            this_alex_fluo,
        ]

        # Extend rows
        nans = [np.nan] * df_events.shape[1]
        while df_events.shape[0] < this_time:
            df_events.loc[df_events.shape[0]] = nans

        # Save fluo values
        df_events.iloc[0:this_time, df_col] = this_donor_fluo
        df_events.iloc[0:this_time, df_col + 1] = this_acceptor_fluo
        df_events.iloc[0:this_time, df_col + 2] = this_alex_fluo
        df_col += 3

        # PER EVENT
        df_this_event = df_separate_events[eidx - 1]
        df_event_col = df_this_event.shape[1]

        # Create columns
        df_this_event[df_event_col] = df_this_event[df_event_col + 1] = df_this_event[
            df_event_col + 2
        ] = np.nan

        # Extend rows
        nans = [np.nan] * df_this_event.shape[1]
        while df_this_event.shape[0] < this_time:
            df_separate_events[eidx - 1] = df_this_event.append(nans)
            df_this_event = df_separate_events[eidx - 1]

        # Save fluo values
        df_this_event.iloc[0:this_time, df_event_col] = this_donor_fluo
        df_this_event.iloc[0:this_time, df_event_col + 1] = this_acceptor_fluo
        df_this_event.iloc[0:this_time, df_event_col + 2] = this_alex_fluo

        # PER EVENT, EbFret like
        ebfret_like_df = pd.DataFrame(
            {
                "id": float(global_eb_fret_counter),
                "donor": donor_fluo[estart:eend],
                "acceptor": acceptor_fluo[estart:eend],
                "alex": alex_fluo[estart:eend],
                "donor_state": donor_state[estart:eend],
                "acceptor_state": acceptor_state[estart:eend],
                "alex_state": alex_state[estart:eend],
                "time": list(range(estart, estart + len(acceptor_state[estart:eend]))),
            }
        )
        df_ebfret_separate_events[eidx - 1] = df_ebfret_separate_events[
            eidx - 1
        ].append(ebfret_like_df)
        global_eb_fret_counter += 1


    return (
        event_plots,
        global_stats,
        df_traces_events,
        df_events,
        df_ebfret_separate_events,
        df_separate_events,
        df_events_information,
    )


def get_dwell_times(idxs_a, idxs_b, idxs_c, idxs_d, idxs_e, df_events_information):
    """
    For each state of an event type, it gets the duration of that event.

    :param idxs_a: Unique ids for events of type A.
    :param idxs_a: Unique ids for events of type B.
    :param idxs_a: Unique ids for events of type C.
    :param df_events_information: All found events' information.
    :return: Tuple with all data found and the broken down information for each
        trace.
    """
    Ad = []
    As = []

    Bd = []
    B1s = []
    B2s = []

    Cd = []
    Cs = []

    Ed = []
    Es = []

    Dd = []
    D1s = []
    D2s = []

    As_t = []
    Bs_t = []
    Cs_t = []
    Es_t = []
    Ds_t = []

    # Parse each row of the events without repeating traces already seen.
    traces_seen = set()
    for _, row in df_events_information.iterrows():
        trace = row["Trace"]
        times_acceptor, acceptor = row["TimesAcceptor"]
        times_donor, donor = row["TimesDonor"]

        if row["EventType"] in idxs_a:
            # Count donor up at first
            donor_pos = 0
            time = 0
            initial_donor = donor[donor_pos]
            while donor[donor_pos] == initial_donor:
                time += times_donor[donor_pos]
                donor_pos += 1

            if trace not in traces_seen:
                traces_seen.add(trace)
                time = np.nan

            Ad.append(time)

            # Seek acceptor change first
            acceptor_pos = 0
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
            time = 0
            initial_donor = donor[0]
            initial_donor = donor[donor_pos]
            while donor[donor_pos] == initial_donor:
                time += times_donor[donor_pos]
                donor_pos += 1

            if trace not in traces_seen:
                traces_seen.add(trace)
                time = np.nan

            Cd.append(time)

            time = 0
            initial_pos = donor_pos
            while donor[donor_pos] == donor[initial_pos]:
                time += times_donor[donor_pos]
                donor_pos += 1

            Cs.append(time)
            Cs_t.append(trace)

        elif row["EventType"] in idxs_e:
            # Seek donor change first
            donor_pos = 0
            time = 0
            initial_donor = donor[donor_pos]
            while donor[donor_pos] == initial_donor:
                time += times_donor[donor_pos]
                donor_pos += 1

            if trace not in traces_seen:
                traces_seen.add(trace)
                time = np.nan

            Ed.append(time)

            time = 0
            initial_pos = donor_pos
            while donor[donor_pos] == donor[initial_pos]:
                time += times_donor[donor_pos]
                donor_pos += 1

            Es.append(time)
            Es_t.append(trace)

        elif row["EventType"] in idxs_d:
            times_aa, aa = row["TimesAA"]
            num_changes = 0
            for a, b in zip(aa, aa[1:]):
                num_changes += int(a != b)

            assert num_changes in (1, 2), f"AA has {num_changes} > 2"

            # Seek donor change first
            donor_pos = 0
            time = 0
            initial_donor = donor[donor_pos]
            while donor[donor_pos] == initial_donor:
                time += times_donor[donor_pos]
                donor_pos += 1

            if trace not in traces_seen:
                traces_seen.add(trace)
                time = np.nan

            Dd.append(time)

            aa_pos = donor_pos
            if num_changes > 1:
                # Seek AA change first
                aa_seek = 0
                initial_aa = aa[aa_seek]
                while aa[aa_seek] == initial_aa:
                    aa_seek += 1

                aa_pos = max(aa_seek, aa_pos)

            # Compute D1, from aa_pos to first aa change
            d1 = 0
            initial_pos = aa_pos
            while aa[aa_pos] == aa[initial_pos]:
                d1 += times_aa[aa_pos]
                aa_pos += 1

            # Find next donor change
            initial_donor = donor[donor_pos]
            while donor[donor_pos] == initial_donor:
                donor_pos += 1

            # Compute D2, from aa_pos to donor_pos (last change)
            d2 = 0
            for pos in range(aa_pos, donor_pos):
                d2 += times_aa[pos]

            D1s.append(d1)
            D2s.append(d2)
            Ds_t.append(trace)


        elif row["EventType"] in idxs_b:

            # Seek donor change first
            donor_pos = 0
            time = 0
            initial_donor = donor[donor_pos]
            while donor[donor_pos] == initial_donor:
                time += times_donor[donor_pos]
                donor_pos += 1

            if trace not in traces_seen:
                traces_seen.add(trace)
                time = np.nan

            Bd.append(time)

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

    maxlen = max(len(As), len(B1s), len(B2s), len(Cs), len(D1s), len(D2s), len(Es))
    Ad = Ad + [np.nan] * (maxlen - len(Ad))
    As = As + [np.nan] * (maxlen - len(As))
    Bd = Bd + [np.nan] * (maxlen - len(Bd))
    B1s = B1s + [np.nan] * (maxlen - len(B1s))
    B2s = B2s + [np.nan] * (maxlen - len(B2s))
    Cd = Cd + [np.nan] * (maxlen - len(Cd))
    Cs = Cs + [np.nan] * (maxlen - len(Cs))
    Dd = Dd + [np.nan] * (maxlen - len(Dd))
    D1s = D1s + [np.nan] * (maxlen - len(D1s))
    D2s = D2s + [np.nan] * (maxlen - len(D2s))
    Ed = Ed + [np.nan] * (maxlen - len(Ed))
    Es = Es + [np.nan] * (maxlen - len(Es))

    As_t = As_t + [np.nan] * (maxlen - len(As_t))
    Bs_t = Bs_t + [np.nan] * (maxlen - len(Bs_t))
    Cs_t = Cs_t + [np.nan] * (maxlen - len(Cs_t))
    Ds_t = Ds_t + [np.nan] * (maxlen - len(Ds_t))
    Es_t = Es_t + [np.nan] * (maxlen - len(Es_t))

    df_data = pd.DataFrame(
        {"A": As, "B1": B1s, "B2": B2s, "C": Cs, "D1": D1s, "D2": D2s, "E": Es}
    )
    df_traces = pd.DataFrame(
        {
            "At": As_t,
            "Ad": Ad,
            "A": As,
            "Bt": Bs_t,
            "Bd": Bd,
            "B1": B1s,
            "B2": B2s,
            "Ct": Cs_t,
            "Cd": Cd,
            "C": Cs,
            "Dt": Ds_t,
            "Dd": Dd,
            "D1": D1s,
            "D2": D2s,
            "Et": Es_t,
            "Ed": Ed,
            "E": Es,
        }
    )

    return df_data, df_traces
