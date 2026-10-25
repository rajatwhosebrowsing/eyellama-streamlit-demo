"""Temporary EyeLlama Streamlit frontend for an authenticated Colab backend."""

import json
import os
import tempfile
from urllib.parse import urlparse

import streamlit as st
from gradio_client import Client, handle_file


st.set_page_config(page_title="EyeLlama research demo", page_icon="👁️", layout="wide")


def valid_backend_url(value):
    parsed = urlparse(value.strip())
    return (
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.hostname.endswith(".gradio.live")
        and not parsed.username
        and not parsed.password
        and parsed.path in ("", "/")
        and not parsed.query
        and not parsed.fragment
    )


def as_object(value):
    return json.loads(value) if isinstance(value, str) else value


def temporary_upload(uploaded):
    suffix = os.path.splitext(uploaded.name)[1].lower()
    if suffix not in (".jpg", ".jpeg", ".png"):
        raise ValueError("Use a JPG or PNG image")
    with tempfile.NamedTemporaryFile(prefix="eyellama_", suffix=suffix, delete=False) as file:
        file.write(uploaded.getvalue())
        return file.name


def connect(url, username, password):
    if not valid_backend_url(url):
        raise ValueError("Paste the HTTPS *.gradio.live link printed by Colab")
    if not username or not password:
        raise ValueError("Enter the username and password printed by Colab")
    return Client(url.strip(), auth=(username, password), verbose=False)


for key, default in (
    ("client", None),
    ("backend_url", ""),
    ("result", None),
    ("chat_history", []),
):
    if key not in st.session_state:
        st.session_state[key] = default

st.title("EyeLlama research demo")
st.warning(
    "Research demonstration only. This is not a diagnosis or clinical triage tool. "
    "Use de-identified images and questions only. Model scores are not medical confidence."
)

with st.sidebar:
    st.header("Connect to Colab")
    st.caption("Keep the Colab backend notebook running throughout the practical.")
    backend_url = st.text_input("Colab Gradio link", placeholder="https://...gradio.live")
    username = st.text_input("Backend username", value="demo")
    password = st.text_input("Backend password", type="password")
    if st.button("Connect", use_container_width=True):
        try:
            st.session_state.client = connect(backend_url, username, password)
            st.session_state.backend_url = backend_url.strip()
            st.session_state.result = None
            st.session_state.chat_history = []
            st.success("Connected to Colab")
        except Exception as exc:
            st.session_state.client = None
            st.error(f"Could not connect: {exc}")
    if st.session_state.client is not None:
        st.success("Colab backend connected")
        st.caption("If Colab restarts, reconnect with its new link and password.")
    st.divider()
    st.caption("The original Gradio page at this link remains your backup demo.")

if st.session_state.client is None:
    st.info("Start the Colab backend notebook, then connect using its link and password.")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.subheader("Analyze an eye image")
    with st.form("analysis_form"):
        uploaded = st.file_uploader("De-identified eye image", type=["jpg", "jpeg", "png"])
        patient_text = st.text_area(
            "English patient question for fusion (optional)",
            max_chars=1500,
            help="Leave blank to test the image-only fallback.",
        )
        method = st.selectbox("Fusion method", ["concatenation", "gated", "late", "image_only"])
        floaters = st.selectbox("Sudden new floaters", ["Unknown", "Yes", "No"])
        flashes = st.selectbox("Flashes of light", ["Unknown", "Yes", "No"])
        curtain = st.selectbox("Curtain or shadow over vision", ["Unknown", "Yes", "No"])
        submitted = st.form_submit_button("Analyze image", type="primary")

    if uploaded is not None:
        st.image(uploaded, caption="Uploaded research image", use_container_width=True)

    if submitted:
        if uploaded is None:
            st.error("Upload an image first")
        elif uploaded.size > 8 * 1024 * 1024:
            st.error("Choose an image smaller than 8 MB")
        else:
            image_path = None
            try:
                image_path = temporary_upload(uploaded)
                with st.spinner("Running image analysis on Colab..."):
                    result = st.session_state.client.predict(
                        handle_file(image_path),
                        patient_text,
                        method,
                        floaters,
                        flashes,
                        curtain,
                        api_name="/analyze_api",
                    )
                result = as_object(result)
                if not isinstance(result, dict) or "top_predictions" not in result:
                    raise RuntimeError("The backend returned an unexpected result")
                st.session_state.result = result
                st.session_state.chat_history = []
                st.success("Analysis complete")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
            finally:
                if image_path and os.path.isfile(image_path):
                    os.unlink(image_path)

with right:
    st.subheader("Model output")
    result = st.session_state.result
    if result is None:
        st.caption("Run an analysis to see model scores and symbolic rules.")
    else:
        st.caption(
            f"Method used: {result.get('method_used', 'unknown')} | "
            f"Text available: {result.get('text_available', False)}"
        )
        st.markdown("**Top model scores (not clinical confidence)**")
        for label, score in result.get("top_predictions", []):
            st.write(f"{label}: {float(score):.2%}")
        st.markdown("**Symbolic rule trace**")
        trace = result.get("symbolic", {}).get("trace", [])
        if trace:
            for entry in trace:
                message = f"{entry.get('rule_id', 'RULE')}: {entry.get('message', '')}"
                if entry.get("rule_id") == "RETINAL_WARNING":
                    st.error(message)
                else:
                    st.info(message)
        else:
            st.info("No implemented rule fired. This does not imply safety.")
        st.caption(result.get("warning", ""))

st.divider()
st.subheader("Follow-up chat")
st.caption(
    "Qwen sees the saved prediction and rule trace, not the image pixels. "
    "It may still make mistakes."
)
for question, answer in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)

with st.form("chat_form"):
    question = st.text_input(
        "Your follow-up question",
        placeholder="What does the INCOMPLETE_SYMPTOMS rule mean?",
        max_chars=1500,
    )
    asked = st.form_submit_button("Ask about this result")

if asked:
    if st.session_state.result is None:
        st.error("Analyze an image before asking a follow-up question")
    elif not question.strip():
        st.error("Type a question first")
    else:
        try:
            with st.spinner("Qwen is generating a reply on Colab. The first reply may take several minutes..."):
                response = st.session_state.client.predict(
                    question,
                    st.session_state.result,
                    st.session_state.chat_history,
                    api_name="/answer_api",
                )
            response = as_object(response)
            if not isinstance(response, dict) or "history" not in response:
                raise RuntimeError("The backend returned an unexpected chat response")
            st.session_state.chat_history = response["history"]
            st.rerun()
        except Exception as exc:
            st.error(f"Chat failed: {exc}")
