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
def _fmt_price(price, currency: str) -> str:
    """'8% of monthly rent' stays as-is; numeric 1800 becomes 'NPR 1,800'."""
    if isinstance(price, (int, float)):
        return f"{currency} {price:,.0f}" if float(price).is_integer() else f"{currency} {price:,}"
    return str(price)

if "products" in _data:
    @mcp.tool()
    def search_products(category: str) -> str:
        """Search inventory by product category.
        """
        data = load_data()
        currency = data.get("currency", "$")
        categories = sorted(set(p.get("category", "") for p in data["products"]) - {""})
        matches = [
            p for p in data.get("products", [])
            if p.get("category") == category.lower()
        ]
        if not matches:
            return f"No products found in '{category}'. Available categories: {', '.join(categories)}"
        lines = "\n".join(f"- {p['name']}: {_fmt_price(p['price'], currency)}" for p in matches)
        return f"Products in '{category}':\n{lines}"

if "services" in _data:
    @mcp.tool()
    def search_services(category: str) -> str:
        """Search services by category."""
        data = load_data()
        currency = data.get("currency", "$")
        categories = sorted(set(s.get("category", "") for s in data["services"]) - {""})
        matches = [
            s for s in data.get("services", [])
            if s.get("category") == category.lower()
        ]
        # Some verticals (e.g. property management) don't categorize services
        # at all. Without this, the tool ALWAYS returned "no services found"
        # with an empty category list. If nothing is categorized, just list
        # everything.
        if not matches and not categories:
            matches = data.get("services", [])
        if not matches:
            return f"No services found in '{category}'. Available categories: {', '.join(categories)}"
        lines = "\n".join(
            f"- {s['name']}: {_fmt_price(s['price'], currency)}"
            + (f" ({s['duration']})" if s.get("duration") else "")
            + (f" — {s['description']}" if s.get("description") else "")
            for s in matches
        )
        return f"Services in '{category}':\n{lines}"

if "faq" in _data:
    # Words that carry no meaning for matching. The old matcher fired if ANY
    # word of a stored question appeared in the user's question — "do", "you",
    # "how" match everything, so it returned the FIRST FAQ regardless of what
    # was actually asked.
    _STOPWORDS = {
        "a", "an", "the", "do", "does", "did", "you", "your", "we", "our", "i",
        "is", "are", "was", "were", "what", "how", "much", "can", "will",
        "of", "for", "to", "in", "on", "at", "with", "and", "or", "any",
        "there", "it", "my", "me", "long", "need", "offer", "have", "if",
    }

    @mcp.tool()
    def get_faq(question: str) -> str:
        """Answer common customer questions about the business."""
        data = load_data()
        q_words = {
            w.strip("?.,!") for w in question.lower().split()
        } - _STOPWORDS
        best_score, best_answer = 0, None
        for item in data["faq"]:
            item_words = {
                w.strip("?.,!") for w in item["q"].lower().split()
            } - _STOPWORDS
            score = len(q_words & item_words)
            if score > best_score:
                best_score, best_answer = score, item["a"]
        # Require at least one MEANINGFUL overlapping word; otherwise punt
        # to the contact number instead of guessing.
        if best_answer and best_score >= 1:
            return best_answer
        return f"For that question please contact us at {data['contact']}."
    
        
if __name__ == "__main__":
    mcp.run()
