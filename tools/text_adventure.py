import random
import time


def text_based_adventure():
    rooms = {
        "Entrance": {
            "description": "You are at the entrance of an ancient temple.",
            "exits": ["Hall", "Garden"],
            "item": None,
        },
        "Hall": {
            "description": "A grand hall with torches lighting the way.",
            "exits": ["Entrance", "Chamber", "Kitchen"],
            "item": "torch",
        },
        "Garden": {
            "description": "A serene garden with a fountain in the center.",
            "exits": ["Entrance", "Library"],
            "item": "key",
        },
        "Chamber": {
            "description": "A dark chamber with a mysterious aura.",
            "exits": ["Hall"],
            "item": "treasure",
        },
        "Kitchen": {
            "description": "An old kitchen with rusted utensils.",
            "exits": ["Hall"],
            "item": "knife",
        },
        "Library": {
            "description": "A dusty library filled with ancient books.",
            "exits": ["Garden"],
            "item": "book",
        },
    }

    inventory = []
    current_room = "Entrance"
    path = [current_room]
    log = []

    log.append(f"Starting adventure in the {current_room}.")
    log.append(rooms[current_room]["description"])

    while True:
        if rooms[current_room]["item"]:
            item = rooms[current_room]["item"]
            inventory.append(item)
            log.append(f"You found a {item}!")
            rooms[current_room]["item"] = None

        if "treasure" in inventory:
            log.append(
                "Congratulations! You found the treasure and won the game!"
            )
            break

        current_room = random.choice(rooms[current_room]["exits"])
        if current_room not in path:
            path.append(current_room)
        log.append(f"Moving to the {current_room}.")
        log.append(rooms[current_room]["description"])
        time.sleep(1)

    return "\n".join(log)


# Run the adventure
adventure_log = text_based_adventure()
print(adventure_log)
