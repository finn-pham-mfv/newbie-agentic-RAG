import json
import asyncio
from datetime import datetime, timezone
from graphiti_core.nodes import EpisodeType
from graphiti_client import GraphitiClient


EPISODES = [
    {
        "content": "Claude is the flagship AI assistant from Anthropic. It was previously "
        "known as Claude Instant in its earlier versions.",
        "type": EpisodeType.text,
        "description": "AI podcast transcript",
    },
    {
        "content": "As an AI assistant, Claude has been available since December 15, 2022 – Present",
        "type": EpisodeType.text,
        "description": "AI podcast transcript",
    },
    {
        "content": {
            "name": "GPT-4",
            "creator": "OpenAI",
            "capability": "Multimodal Reasoning",
            "previous_version": "GPT-3.5",
            "training_data_cutoff": "April 2023",
        },
        "type": EpisodeType.json,
        "description": "AI model metadata",
    },
    {
        "content": {
            "name": "GPT-4",
            "release_date": "March 14, 2023",
            "context_window": "128,000 tokens",
            "status": "Active",
        },
        "type": EpisodeType.json,
        "description": "AI model metadata",
    },
]


async def ingest_episodes():
    graphiti = await GraphitiClient().create_client(
        clear_existing_graphdb_data=True,
        max_coroutines=1,
    )

    try:
        # Episodes are the primary units of information in Graphiti.
        # They can be text or structured JSON
        # and are automatically processed to extract entities & relationships.

        # Add episodes to the graph
        for i, episode in enumerate(EPISODES):
            await graphiti.add_episode(
                name=f"AI Agents Unleashed {i}",
                episode_body=(
                    episode["content"]
                    if isinstance(episode["content"], str)
                    else json.dumps(episode["content"])
                ),
                source=episode["type"],
                source_description=episode["description"],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: AI Agents Unleashed {i} ({episode["type"].value})')
    finally:
        # Always close the connection to Neo4j when finished to properly release resources
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(ingest_episodes())
