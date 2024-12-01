import streamlit as st
import plotly.graph_objects as go
from llm.llm import LLMRouter, load_data, modify_data, train_model, answer_text, answer_visual
import random

# Initialize router
if 'router' not in st.session_state:
    st.session_state.router = LLMRouter()
    st.session_state.router.register_function("load_data", load_data)
    st.session_state.router.register_function("modify_data", modify_data)
    st.session_state.router.register_function("train_model", train_model)
    st.session_state.router.register_function("answer_text", answer_text)
    st.session_state.router.register_function("answer_visual", answer_visual)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("BMS Data Science Assistant")
st.markdown("Welcome to the BMS Data Science Assistant! Ask me anything about drug consumption data.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # if "classification" in message:
        #     st.text(f"Function called: {message['classification']['function']}")
        #     st.text(f"Reasoning: {message['classification']['reasoning']}")
        st.markdown(message["content"])
        if "figure" in message:
            st.plotly_chart(message["figure"], use_container_width=True, key=random.randint(0, 1000))

# Handle user input
if prompt := st.chat_input("Ask about drug consumption data"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            classification = st.session_state.router.classify_query(prompt)
            result = st.session_state.router.route_and_execute(prompt)

            response = {
                "role": "assistant",
                "classification": classification
            }

            if isinstance(result, go.Figure):
                st.plotly_chart(result, use_container_width=True)
                response["content"] = "Here's the visualization you requested."
                response["figure"] = result
            else:
                st.write(result)
                response["content"] = str(result)

            st.session_state.messages.append(response)

        except Exception as e:
            raise e
            st.error(f"Error: {str(e)}")
