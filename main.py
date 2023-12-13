import streamlit as st
from streamlit_chat import message as st_message
import langchain_helper as lch
@st.cache_resource()

class QAApplication:
    def __init__(self):
        self.db = lch.create_db_from_documents()

    def generate_answer(self, openai_api_key):
        request = st.session_state.request
        response = lch.get_response_from_query(
            self.db, request, openai_api_key=openai_api_key, k=4)

        st.session_state.messages.append({"role": "user", "content": request})
        st.session_state.messages.append({"role": "assistant", "content": response})

    def run_app(self):
        st.title("Ask me anything about Spain Immigration:")

        if "history" not in st.session_state:
            st.session_state.history = []
        with st.sidebar:
          openai_api_key = st.sidebar.text_input(
              label="OpenAI API Key",
              key="openai_api_key",
              max_chars=100,
              type="password"
              )
          "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

        if st.button("Clear"):
            st.session_state.history = []

        prompt = st.chat_input("", key="request")

        if not openai_api_key:
          st.info("Please add your OpenAI API key to continue.")
          st.stop()

        if prompt and openai_api_key:
            self.generate_answer(openai_api_key)

        for message in st.session_state.messages:
          print(message)
          with st.chat_message(message["role"]):
              st.markdown(message["content"])

if __name__ == "__main__":
    if "messages" not in st.session_state:
      st.session_state.messages = []
    qa = QAApplication()
    qa.run_app()