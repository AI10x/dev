import shelve
from keyboard import keyboard


agg = shelve.open("/home/emmanuel/rec.db", writeback=True)



aggregate = []
if __name__ == "__main__":
    if 'recorder' in agg:
        recorded_events = agg['recorder']
        print(f"Found {len(recorded_events)} events in rec.db.")
        
        # Iterate through each KeyboardEvent object in the list
        N_transitions_index = []
        for i, event in enumerate(recorded_events):
            # Check if the event is for the 'enter' key (either press or release)
            if event.name == 'enter':
                N_transitions_index.append(i)
                print(f"Found 'enter' key press event: {event}")

        # Add start and end boundaries for slicing
        slice_indices = [0] + N_transitions_index + [len(recorded_events)]

        shifted = []
        # Create pairs of indices to slice the recorded_events list.
        # This ensures all segments are captured.
        for start, end in zip(slice_indices, slice_indices[1:]):
            # For segments after the first one, we add 1 to the start index.
            # This excludes the 'enter' event from the beginning of the next segment.
            start_index = start + 1 if start != 0 else 0
            
            shifting = recorded_events[start_index:end]
            shifted.append(shifting)

        for xyz in shifted:
            print(xyz)

    else:
        print("No 'recorder' data found in rec.db. Run rec.py to record keystrokes.")


agg.close()