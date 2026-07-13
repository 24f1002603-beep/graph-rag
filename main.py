def call_aipipe_llm(prompt: str) -> dict:
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {
            "type": "json_object"
        }
    }

    try:
        response = requests.post(
            AIPIPE_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        print("Status Code:", response.status_code)
        print("Response Body:", response.text)

        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return json.loads(content)

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"AIPipe request failed: {str(e)}"
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON."
        )
