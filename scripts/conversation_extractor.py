import re
import json
import subprocess
from pathlib import Path

def get_conversation_tweets(root_id, max_items=25):
    """
    Calls `snscrape twitter-conversation <root_id> --jsonl`,
    yields up to max_items JSON dicts (including the root tweet).
    """
    cmd = ["snscrape", "twitter-conversation", root_id, "--jsonl"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    items = []
    for line in proc.stdout:
        if len(items) >= max_items:
            break
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    proc.stdout.close()
    proc.wait()
    return items

def build_thread_tree(items, root_id, max_replies=24):
    """
    Given a list of tweet‐dicts (each with 'id', 'content', 'inReplyToTweetId', 'createdAt'),
    build a nested tree under the root_id, including up to max_replies descendants.
    """
    
    nodes = {
        str(item["id"]): {
            "id": str(item["id"]),
            "content": item.get("content", ""),
            "children": []
        }
        for item in items
    }
    
    items_sorted = sorted(items, key=lambda x: x.get("createdAt", ""))

    count = 0
    for item in items_sorted:
        tid = str(item["id"])
        pid = str(item.get("inReplyToTweetId") or "")
        # skip root itself
        if tid == root_id:
            continue
        
        if pid in nodes and count < max_replies:
            nodes[pid]["children"].append(nodes[tid])
            count += 1

    
    return nodes[root_id]

def main():
    url = input("Enter the URL of the parent tweet: ").strip()
    m = re.search(r"twitter\.com/[^/]+/status/(\d+)", url)
    if not m:
        print("❌ Couldn't parse a tweet ID from that URL.")
        return

    root_id = m.group(1)
    print(f"🔍 Fetching conversation for tweet ID {root_id}…")
    
    items = get_conversation_tweets(root_id, max_items=25)
    if not items:
        print("⚠️ No tweets returned. Is the tweet ID correct and public?")
        return

    tree = build_thread_tree(items, root_id, max_replies=24)

    out_path = Path("thread.json")
    out_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    num_replies = len(tree["children"])
    print(f"✅ Extracted {num_replies} replies; tree saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
