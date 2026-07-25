import streamlit as st
from combined_agent import agent, config

st.set_page_config(page_title="Weather Research Assistant", page_icon="🌧️")
st.title("Weather Intelligence Research Assistant")
st.caption("Ask about monsoon forecasting research, or request a live precipitation prediction.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question or request a prediction..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
            answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})