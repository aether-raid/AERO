import asyncio
from aero.experiment_designer import run_experiment_designer
from dotenv import load_dotenv
load_dotenv()

async def run_main():
    user_input = input("Enter your research plan: ")
    # Await to get the async generator object
    gen = await run_experiment_designer(user_input, stream=True)
    final_state = None
    async for msg in gen:
        final_state = msg
        if isinstance(msg, dict) and "stream_mode" in msg:
            if msg["stream_mode"] == "update":
                print(f"\n=== NODE: {msg['data']} ===")
            elif msg["stream_mode"] == "custom":
                print(msg["data"])
        elif not isinstance(msg, dict):
            print(msg)

    # Write only the final state (design and code) to a Markdown file
    if isinstance(final_state, dict):
        design = final_state.get("design", "")
        code = final_state.get("code", "")
        with open("experiment_output.md", "w", encoding="utf-8") as f:
            f.write("# Experiment Design\n\n")
            f.write(design.strip() + "\n\n")
            f.write("# Generated Code\n\n")
            f.write("```python\n")
            f.write(code.strip() + "\n")
            f.write("```\n")
        print("Final state written to experiment_output.md")
    else:
        print("No valid final state to write.")
        
if __name__ == "__main__":
    asyncio.run(run_main())