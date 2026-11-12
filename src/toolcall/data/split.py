"""Stable query normalization and grouped split assignment."""
from collections import Counter, defaultdict
import random
import unicodedata

def query_key(text):
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text).split())



def make_splits(records, seed=42):
    groups = defaultdict(list)
    for row in records:
        q = query_key(row["audio_text"])
        groups[q if q else f"\0{row['id']}"] .append(row)
    # Large groups first. Seeded tie-breaking, independent of input order.
    rng = random.Random(seed)
    group_list = [sorted(g, key=lambda r: r["id"]) for _, g in sorted(groups.items())]
    rng.shuffle(group_list)
    group_list.sort(key=len, reverse=True)
    fractions = {"train": .8, "val": .1, "test": .1}
    outputs = {key: [] for key in fractions}
    category_total = Counter(min(len(r["label"]["tool_calls"]), 2) for r in records)
    category_counts = {key: Counter() for key in fractions}
    for group in group_list:
        categories = Counter(min(len(r["label"]["tool_calls"]), 2) for r in group)
        def cost(key):
            # Normalized change in squared size and scenario deviations.
            def delta(current, increment, target):
                return ((current + increment - target) ** 2 - (current - target) ** 2) / max(target, 1)
            return delta(len(outputs[key]), len(group), fractions[key] * len(records)) + sum(delta(category_counts[key][c], n, fractions[key] * category_total[c]) for c, n in categories.items())
        chosen = min(fractions, key=cost)
        outputs[chosen].extend(r["id"] for r in group)
        category_counts[chosen].update(categories)
    return {key: sorted(value) for key, value in outputs.items()}
