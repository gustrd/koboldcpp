import json, re

def normalize(obj):
    if "type" in obj and "function" in obj: return obj
    if "name" in obj:
        args = obj.get("arguments", obj.get("parameters", {}))
        return {"type":"function","function":{"name":obj["name"],"arguments":args}}
    if "function" in obj and "name" in obj["function"]:
        f = obj["function"]
        return {"type":"function","function":{"name":f["name"],"arguments":f.get("arguments",{})}}
    return obj

# Test Case 2: Raw OpenAI JSON Array (the new failing case)
text2 = """[
{"id":"call_001","type":"function","function":{"name":"read","arguments":{"path":"/root/.openclaw/workspace/SOUL.md"}}},
{"id":"call_001","type":"function","function":{"name":"read","arguments":{"path":"/root/.openclaw/workspace/USER.md"}}}
]"""

print("=== Test Case 2: Raw OpenAI JSON array ===")
parsed = json.loads(text2)
parsed = [normalize(o) for o in parsed]
parsed = [tc for tc in parsed if tc.get("function",{}).get("name")]
for tc in parsed:
    a = tc["function"].get("arguments")
    if a and not isinstance(a, str):
        tc["function"]["arguments"] = json.dumps(a)
print(f"Tools found: {len(parsed)}")
for tc in parsed:
    print(f"  - name={tc['function']['name']} | args={tc['function']['arguments']}")

# Test Case 1: Custom tag
text1 = '[TOOL_CALLS]message[ARGS]{"action": "send", "channel": "telegram"}'
pattern = r'\[TOOL_CALLS\](.*?)\[ARGS\](\{.*?\})'
print("\n=== Test Case 1: [TOOL_CALLS] tag format ===")
results = []
for m in re.finditer(pattern, text1, re.DOTALL):
    args = json.loads(m.group(2))
    results.append({"type":"function","function":{"name":m.group(1).strip(),"arguments":m.group(2)}})
print(f"Tools found: {len(results)}")
for tc in results:
    print(f"  - name={tc['function']['name']} | args={tc['function']['arguments']}")

# Test Case 4: Plain text - must NOT produce tool calls
text4 = "Esta e uma resposta normal do modelo sem nenhuma chamada de ferramenta."
print("\n=== Test Case 4: Plain text (should produce 0 tools) ===")
try:
    parsed4 = json.loads(text4)
except:
    parsed4 = []
parsed4 = [normalize(o) for o in parsed4] if isinstance(parsed4, list) else []
parsed4 = [tc for tc in parsed4 if tc.get("function",{}).get("name")]
print(f"Tools found: {len(parsed4)} (expected: 0) {'OK' if len(parsed4)==0 else 'FAIL'}")
