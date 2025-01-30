import re
import time
import streamlit as st
from groq import Groq
import webbrowser

# -----------------------------
# Alapbeállítások
# -----------------------------
st.set_page_config(
    page_title="DeepSeek Magyarul csoport",
    page_icon="💬",
    layout="wide",
)

# -----------------------------
# Custom CSS
# -----------------------------
custom_css = """
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.chat-bubble {
    padding: 10px 15px;
    margin-bottom: 8px;
    border-radius: 8px;
    max-width: 80%;
    line-height: 1.4;
}
.user-bubble {
    background-color: #0084ff20;
    border: 1px solid #0084ff;
    color: #111;
    align-self: flex-end;
}
.assistant-bubble {
    background-color: #fafafa;
    border: 1px solid #ddd;
    color: #111;
    align-self: flex-start;
}
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.css-18e3th9 {
    padding: 1rem 1rem 1rem 2rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# -----------------------------
# Segédfüggvények
# -----------------------------
def parse_reasoning(raw_text: str):
    """ Kinyeri a <think> blokkok tartalmát és törli őket a végső válaszból. """
    reasoning_parts = re.findall(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    reasoning_text = "\n".join(reasoning_parts)
    cleaned_answer = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    return cleaned_answer.strip(), reasoning_text.strip()


# -----------------------------
# Inicializálás
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""  # Felhasználói input tárolása a sessionben


# -----------------------------
# Oldalsáv (API-kulcs, model, stb.)
# -----------------------------
with st.sidebar:
    st.title("Beállítások")

    api_key = st.text_input("Groq API-kulcs", placeholder="Írd be az API-kulcsodat...", type="password")

    model_options = ["deepseek-r1-distill-llama-70b"]
    model_name = st.selectbox("Modell kiválasztása:", model_options, index=0)

    temperature = st.slider("Hőmérséklet", 0.0, 1.5, 0.6, 0.1)
    top_p = st.slider("Top_p", 0.0, 1.0, 0.95, 0.05)
    max_tokens = st.number_input("Max tokenszám", value=20000, min_value=1, max_value=25000, step=1)

    streaming = st.checkbox("Streaming mód", value=True)

    # Új chat indítása -> töröljük az üzeneteket és újrarendereljük az oldalt
    if st.button("🗑️ Új chat indítása"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Groq fiók regisztráció"):
        webbrowser.open_new_tab("https://console.groq.com/keys")

    st.write("Kérlek vedd figyelembe, hogy ez egy minimalista fejlesztés. A fejlesztés csak tesztelési célokat szolgál. ")
    st.write("""


""")


# -----------------------------
# Főcím
# -----------------------------
st.title("DeepSeek Magyarul csoport")


# -----------------------------
# Inputmező és küldés gomb (Enter gombbal is elküldhető)
# -----------------------------
user_input = st.text_input(
    "Írd be a kérdésed...",
    value=st.session_state.user_input,  # A session state alapértéke
    key="input_box",
    on_change=lambda: st.session_state.update(user_input=st.session_state.input_box)
)

if st.button("Küldés"):
    if st.session_state.user_input.strip():
        st.session_state.messages.append({"role": "user", "content": st.session_state.user_input.strip()})
        st.session_state.user_input = ""  # Töröljük az inputmező tartalmát
        st.rerun()


# -----------------------------
# API-hívás, ha van új user-üzenet
# -----------------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if not api_key or api_key.strip() == "":
        st.warning("Kérlek, adj meg érvényes Groq API-kulcsot!")
        st.stop()

    try:
        client = Groq(api_key=api_key.strip())

        messages_for_api = [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
        ]

        # Streaming mód esetén folyamatos frissítés
        if streaming:
            response_placeholder = st.empty()
            reasoning_placeholder = st.empty()
            raw_accumulated = ""

            completion = client.chat.completions.create(
                model=model_name,
                messages=messages_for_api,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_tokens,
                reasoning_format="raw",
                stream=True
            )

            for chunk in completion:
                delta_text = chunk.choices[0].delta.content or ""
                raw_accumulated += delta_text

                partial_answer, partial_rsn = parse_reasoning(raw_accumulated)

                response_placeholder.markdown(f"**Válasz (folyamatos betöltés):**\n{partial_answer}")
                if partial_rsn:
                    reasoning_placeholder.markdown(f"**Gondolkodás**:\n```\n{partial_rsn}\n```")

                time.sleep(0.05)

            response_placeholder.empty()
            reasoning_placeholder.empty()
            final_answer, final_reason = parse_reasoning(raw_accumulated)

        else:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages_for_api,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_tokens,
                reasoning_format="raw",
                stream=False
            )

            raw_text = completion.choices[0].message.content
            final_answer, final_reason = parse_reasoning(raw_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "reasoning": final_reason
        })

        st.rerun()  # Frissítjük az oldalt, hogy az új válasz megjelenjen

    except Exception as e:
        st.error(f"Hiba történt a Groq API-hívás során: {e}")


# -----------------------------
# Beszélgetés megjelenítése
# -----------------------------
chat_container = st.container()
for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_container.markdown(f'<div class="chat-container"><div class="chat-bubble user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        chat_container.markdown(f'<div class="chat-container"><div class="chat-bubble assistant-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        if msg.get("reasoning"):
            with chat_container.expander("Gondolkodás (lenyitható)", expanded=False):
                st.markdown(f"<pre>{msg['reasoning']}</pre>", unsafe_allow_html=True)


st.write("---")
st.write("**Megjegyzés:** Ha észrevételetek van, vagy megjegyzéstek, nyugodtan írjatok a csoportba: https://www.facebook.com/groups/deepseekmagyarul/")
# -----------------------------
# Lábléc: Groq logó és link
# -----------------------------
st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <a href="https://groq.com" target="_blank" rel="noopener noreferrer">
            <img src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg" 
                 alt="Powered by Groq for fast inference."
                 style="width: 200px;">
        </a>
    </div>
""", unsafe_allow_html=True)

