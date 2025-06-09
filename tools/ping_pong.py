import time
import os
import keyboard

# Initialize the game variables
paddle_position = 3
ball_position_x = 10
ball_position_y = 5
ball_direction_x = -1
ball_direction_y = 1
score = 0


# Function to clear the terminal screen
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# Function to draw the game
def draw_game():
    clear_screen()
    # Draw the top and bottom boundaries
    print("+----------------------+")
    # Draw the game area
    for y in range(10):
        line = "|"
        for x in range(20):
            if x == 0 and y == paddle_position:
                line += "|"
            elif x == ball_position_x and y == ball_position_y:
                line += "O"
            else:
                line += " "
        line += "|"
        print(line)
    # Draw the bottom boundary
    print("+----------------------+")
    print(f"Score: {score}")


# Main game loop
try:
    while True:
        # Move the ball
        ball_position_x += ball_direction_x
        ball_position_y += ball_direction_y

        # Ball collision with top and bottom
        if ball_position_y <= 0 or ball_position_y >= 9:
            ball_direction_y *= -1

        # Ball collision with paddle
        if ball_position_x <= 0 and ball_position_y == paddle_position:
            ball_direction_x *= -1
            score += 1

        # Ball out of bounds
        if ball_position_x < 0:
            ball_position_x = 10
            ball_position_y = 5
            score -= 1

        # Ball collision with right wall
        if ball_position_x >= 19:
            ball_direction_x *= -1

        # Move the paddle with keyboard input
        if keyboard.is_pressed("up") and paddle_position > 0:
            paddle_position -= 1
        if keyboard.is_pressed("down") and paddle_position < 9:
            paddle_position += 1

        # Draw the game
        draw_game()

        # Control the game speed
        time.sleep(0.2)
except KeyboardInterrupt:
    print("\nGame Over")
