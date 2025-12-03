import json
import os
relations_dir = "relations/"
relations_json_dir = "relations_json/"
all_relations = []
for filename in os.listdir(relations_dir):
    triples = []
    with open(relations_dir+filename, "r", encoding="utf-8") as f:
        text = f.read()
        for line in text.strip().split("\n"):
            if len(line.strip()) > 0:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    cause, relation, effect = parts
                    triples.append({
                        "cause": cause,
                        "relation": relation,
                        "effect": effect
                    })
                    all_relations.append({
                        "cause": cause,
                        "relation": relation,
                        "effect": effect
                    })

    # Final JSON structure
    output = {"triples": triples}

    # Write to JSON file
    with open(relations_json_dir+filename.split(".")[0]+".json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
output = {"triples": all_relations}
with open("all_relations.json",  "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)