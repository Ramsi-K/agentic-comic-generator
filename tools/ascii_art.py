from PIL import Image

# Load the image
img = Image.open("storyboard/test_session_modal_image_gen/content/panel_1.png")

# Resize the image
width, height = 100, 60
img = img.resize((width, height))

# Convert the image to grayscale
img = img.convert("L")

# ASCII characters used to represent the image
ascii_chars = "@%#*+=-:. "

# Convert the image to ASCII art
ascii_str = ""
for y in range(height):
    for x in range(width):
        pixel = img.getpixel((x, y))
        ascii_str += ascii_chars[pixel // 32]
    ascii_str += "\n"

# Print the ASCII art
print(ascii_str)
