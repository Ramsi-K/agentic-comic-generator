import time
import os

# Define ASCII art frames for a simple dancing figure
frames = [
    r"""
          O
         /|\
         / \
    """,
    r"""
          O
         /|\
         / \
          |  O
         / \
    """,
    r"""
          O
         /|\
         / \
        O
         / \
    """,
]


# Function to clear the terminal screen
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# Display each frame in sequence to create an animation
while True:
    for frame in frames:
        clear_screen()
        print(frame)
        time.sleep(0.3)
