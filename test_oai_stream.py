import requests
import json

def test_oai_streaming_raw():
    url = "http://localhost:5001/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": "koboldcpp",
        "messages": [
            {"role": "user", "content": "Help me test this custom streaming feature by saying 'Successfully tested fake streaming' exactly one time."}
        ],
        "stream": True,
        "max_tokens": 100
    }

    print(f"Connecting to {url}...")
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        print("\n--- Raw SSE Stream Output ---")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(f"Received: {decoded_line}")
                
                if decoded_line.startswith("data:"):
                    data_str = decoded_line[5:].strip()
                    if data_str == "[DONE]":
                        print("Stream complete [DONE]")
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk['choices'][0]['delta']
                        if 'content' in delta:
                            print(f"  -> Content: '{delta['content']}'")
                        if 'role' in delta:
                            print(f"  -> Role: {delta['role']}")
                        if chunk['choices'][0].get('finish_reason'):
                            print(f"  -> Finish Reason: {chunk['choices'][0]['finish_reason']}")
                    except Exception as e:
                        print(f"  (Failed to parse JSON: {e})")

    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_oai_streaming_raw()
