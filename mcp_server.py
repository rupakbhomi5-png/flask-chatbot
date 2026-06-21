"""
mcp_server.py — Ramcha MCP Tool Server
Runs as a standalone process. Flask app connects via stdio.
Add new tools here; app_memory.py discovers them automatically."""

import json
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("raj-cassette")

def load_products() -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "products.json"), "r") as f:
        return json.load(f)


@mcp.tool()
def search_products(category: str) -> str:
    """Search Raj Cassette inventory by category.
    Options: remotes, speakers, chargers, batteries, accessories, appliances,
    televisions, storage, tv_boxes, fans, phones, audio, computers, headphones,
    networking, security, lighting."""
    data = load_products()
    matches = [
        p for p in data["products"]
        if p.get("category") == category.lower()
    ]
    if not matches:
        return f"No products found in category '{category}'."
    lines = "\n".join(f"- {p['name']}: NPR {p['price']:,}" for p in matches)
    return f"Products in '{category}':\n{lines}"

if __name__ == "__main__":
    mcp.run()
