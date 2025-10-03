from keyboard import keyboard
import time

# A flag to prevent recursive event triggers.

def on_key_event(event: keyboard.KeyboardEvent):
    """
    This function is called for any keyboard event (press or release).
    If the 'enter' key is pressed, it simulates a 'ctrl+s' key press. Use case - saving page as you navigate the browser
    """


    # We only care about key press ('down') events for our trigger.
    if event.event_type == keyboard.KEY_DOWN:
        print(f"Detected press: '{event.name}'")
        # Check if the trigger key ('enter') was pressed.
        if event.name == 'enter':
            time.sleep(1)
            print("-> 'enter' detected! Simulating 'ctrl+s' key press.")
            keyboard.send('ctrl+s') # Simulate pressing and releasing 'ctrl+s'
            # Return True to block the original 'enter' key from being processed.
            return True








def main():
    """
    Main function to set up the keyboard listener and wait for exit.
    """



    print("Starting key remapper. Press 'enter' to trigger 'ctrl+s'. Press 'esc' to exit.")

    # keyboard.hook() registers a callback for all keyboard events (press and release).
    keyboard.hook(on_key_event)
    

    # keyboard.wait() blocks the program until the specified key is pressed.
    # This keeps your script running to listen for events.
    keyboard.wait('esc')
    print("\n'esc' key pressed. Exiting program.")

if __name__ == "__main__":
    main()
