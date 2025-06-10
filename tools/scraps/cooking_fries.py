import time
import os

# Define ASCII art frames for fries cooking
frames = [
    r"""
    __________
   |          |
   |          |
   |          |
   |          |
   |__________|
    (\_/)
    """,
    r"""
    __________
   | ~~~~~~~~ |
   | ~~~~~~~~ |
   | |  |  |  |
   | |__|__|__|
   |__________|
    (\_/) O
    """,
    r"""
    __________
   | ~~~~~~~~ |
   | ~~~~~~~~ |
   | |  |  |  |
   | |__|__|__|
   |__________|
    (\_/)=
    """,
    r"""
    __________
   | ~~~~~~~~ |
   | ~~~~~~~~ |
   | |  |  |  |
   | |__|__|__|
   |__________|
    (\_/)
    """,
]


# Function to clear the terminal screen
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# Display each frame in sequence to create an animation
while True:
    for frame in frames:
        clear_screen()
        print("Cooking fries...")
        print(frame)
        time.sleep(0.5)
