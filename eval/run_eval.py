import json
from app import retrieve

questions = json.load(open("eval/questions.json"))

hits = 0
rr_scores = []

for q in questions:
    docs = retrieve(q["question"])
    found = [d.metadata.get("page", 0) + 1 for d in docs]
    expected = q["expected_page"]

    if expected:
        ok = bool(set(found) & set(expected))
    else:
        ok = len(docs) == 0

    if ok:
        hits += 1

    # MRR: only for questions with an expected page (negative cases have no rank)
    rr = 0.0
    if expected:
        for position, page in enumerate(found, start=1):
            if page in expected:
                rr = 1.0 / position
                break
        rr_scores.append(rr)

    print(f"{q['id']:>2} {'OK  ' if ok else 'MISS'} | RR {rr:.2f} | expected {expected} | found {found}")

print(f"\nContext Recall: {hits}/{len(questions)} = {hits/len(questions):.0%}")
print(f"MRR (excluding negative cases): {sum(rr_scores)/len(rr_scores):.2f}")
