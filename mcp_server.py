"""
mcp_server.py — Unified MCP Tool Server
Reads business_data.json (or DATA_FILE env var).
Dynamically registers only the tools that match the data key present.
- products key → registers search_products
- services key → registers search_services  
- faq key     → registers get_faq
Flask app discovers all registered tools automatically at startup."""

import json
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("business-bot")

def load_data() -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.environ.get("DATA_FILE", "business_data.json")
    with open(os.path.join(base_dir, data_file), "r") as f:
        return json.load(f)
# Load once at startup - tools are registered based on what keys exists
_data = load_data()
if "products" in _data:
    @mcp.tool()
    def search_products(category: str) -> str:
        """Search inventory by product category.
        """
        data = load_data()
        categories = list(set(p.get("category", "") for p in data["products"]))
        matches = [
            p for p in data.get("products", [])
            if p.get("category") == category.lower()
        ]
        if not matches:
            return f"No products found in '{category}'. Available categories: {', '.join(categories)}"
        lines = "\n".join(f"- {p['name']}: NPR {p['price']:,}" for p in matches)
        return f"Products in '{category}':\n{lines}"

if "services" in _data:
    @mcp.tool()
    def search_services(category: str) -> str:
        """Search services by category."""
        data = load_data()
        categories = list(set(s.get("category", "") for s in data["services"]))
        matches = [
            s for s in data.get("services", [])
            if s.get("category") == category.lower()
        ]
        if not matches:
            return f"No services found in '{category}'. Available categories: {', '.join(categories)}"
        lines = "\n".join(f"- {s['name']}: ${s['price']} ({s['duration']})" for s in matches)
        return f"Services in '{category}':\n{lines}"

if "faq" in _data:
    @mcp.tool()
    def get_faq(question: str) -> str:
        """Answer common customer questions about the business."""
        data = load_data()
        question_lower = question.lower()
        for item in data["faq"]:
            if any(word in question_lower for word in item["q"].lower().split()):
                return item["a"]
        return f"For that question please contact us at {data['contact']}."
    
        
if __name__ == "__main__":
    mcp.run()
