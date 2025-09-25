#!/usr/bin/env python3
"""Streaming test harness for the model researcher workflow."""

import asyncio
import os
import sys
from typing import Optional, Dict, Any

from dotenv import load_dotenv


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.join(ROOT_DIR, "..", "src")
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)


async def main() -> Optional[Dict[str, Any]]:
    """Run the streaming workflow and print key updates."""
    load_dotenv()

    from aero.model_researcher import stream_model_suggestions

    prompt = (
        "Design a streaming anomaly detection pipeline for industrial IoT sensors "
        "that must flag potential equipment failures within 30 seconds."
    )

    print("🤖 Streaming Model Researcher Test")
    print("=" * 60)
    print(f"📝 Prompt: {prompt}")
    print()

    final_result: Optional[Dict[str, Any]] = None
    update_count = 0

    async for update in stream_model_suggestions(prompt):
        update_count += 1
        status = update.get("status") or update.get("current_step")
        if status:
            print(f"➡️  Update {update_count}: {status}")

        if "model_suggestions" in update and update["model_suggestions"].get("model_suggestions"):
            summary = update["model_suggestions"]["model_suggestions"]
            preview = summary[:160] + ("..." if len(summary) > 160 else "")
            print("📄 Suggestion Preview:")
            print(preview)
            print()

        final_result = update

    if not final_result:
        print("❌ No updates received from streaming workflow.")
        return None

    print("✅ Streaming workflow completed!")
    suggestions = final_result.get("model_suggestions", {}).get("model_suggestions")
    if suggestions:
        print("\n📋 Final Suggestions (first 500 chars):")
        print(suggestions[:500] + ("..." if len(suggestions) > 500 else ""))
    else:
        print("⚠️ No final suggestions present in result.")

    if final_result.get("errors"):
        print("\n⚠️ Errors encountered during workflow:")
        for err in final_result["errors"][-3:]:
            print(f"   - {err}")

    return final_result


if __name__ == "__main__":
    asyncio.run(main())
