from brown import create_agent_brown, StoryboardRequest
from bayko import create_agent_bayko
import asyncio


async def main():
    # Step 1: Set up Brown and Bayko
    brown = create_agent_brown()
    bayko = create_agent_bayko()

    # Step 2: Prepare user input
    request = StoryboardRequest(
        prompt="A time-traveling cat battles boredom in a rainy future Seoul.",
        style_preference="studio_ghibli",
        panels=3,
        language="english",
        extras=["narration", "subtitles"],
    )

    # Step 3: Brown processes the request
    message = brown.process_request(request)

    # Step 4: Bayko processes the message
    result = await bayko.process_generation_request(message.to_dict())

    # Step 5: Print results
    print(f"\n✅ Session: {result.session_id}")
    print(f"Status: {result.status.value}")
    for panel in result.panels:
        print(f"\n🖼️ Panel {panel.panel_id}:")
        print(f"Description: {panel.description}")
        print(f"Image: {panel.image_path}")
        print(f"Audio: {panel.audio_path}")
        print(f"Subtitles: {panel.subtitles_path}")
        print(f"Time: {panel.generation_time:.2f}s")
        if panel.errors:
            print(f"⚠️ Errors: {panel.errors}")


if __name__ == "__main__":
    asyncio.run(main())
