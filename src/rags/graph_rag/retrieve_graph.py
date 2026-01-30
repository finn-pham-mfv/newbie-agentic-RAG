import asyncio
from graphiti_client import GraphitiClient


async def retrieve_graph(num_results: int = 10):
    graphiti = await GraphitiClient().create_client(
        clear_existing_graphdb_data=False,
        max_coroutines=1,
    )

    results = await graphiti.search(
        "Which AI assistant is from Anthropic?",
        num_results=num_results,
    )

    return results


if __name__ == "__main__":
    results = asyncio.run(retrieve_graph(num_results=2))
    print(results)
    print("\nSearch Results:")
    for result in results:
        print(f"UUID: {result.uuid}")
        print(f"Fact: {result.fact}")
        if hasattr(result, "valid_at") and result.valid_at:
            print(f"Valid from: {result.valid_at}")
        if hasattr(result, "invalid_at") and result.invalid_at:
            print(f"Valid until: {result.invalid_at}")
        print("---")
