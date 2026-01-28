import streamlit as st
import requests

API_ROOT = "http://dgx0.chapman.edu:12345/v1"
HEADERS = {"Content-Type": "application/json"}

st.title("llama.cpp Chat UI")
st.write("Interact with your remote llama.cpp server via OpenAI-compatible API.")

if "history" not in st.session_state:
    st.session_state.history = []

def get_model_id():
    r = requests.get(f"{API_ROOT}/models", timeout=10)
    r.raise_for_status()
    j = r.json()
    return j["data"][0]["id"]

if "model_id" not in st.session_state:
    try:
        st.session_state.model_id = get_model_id()
    except Exception as e:
        st.session_state.model_id = ""
        st.error(f"Failed to fetch models: {e}")

model_id = st.text_input("Model", st.session_state.model_id)

prompt = st.text_input("You:", "")

if st.button("Send") and prompt.strip():
    messages = []
    for speaker, text in st.session_state.history:
        role = "user" if speaker == "You" else "assistant"
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 2500,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(f"{API_ROOT}/chat/completions", json=body, headers=HEADERS, timeout=600)
        if resp.status_code != 200:
            assistant_msg = f"HTTP {resp.status_code}: {resp.text}"
        else:
            data = resp.json()
            assistant_msg = data["choices"][0]["message"]["content"]
    except Exception as e:
        assistant_msg = f"Error: {e}"

    st.session_state.history.append(("You", prompt))
    st.session_state.history.append(("Assistant", assistant_msg))

for speaker, text in st.session_state.history:
    st.write(f"{speaker}: {text}")