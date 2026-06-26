import streamlit as st

with st.chat_message('user'):
    st.text('Hi')

with st.chat_message('assistant'):
    st.text("You've summoned the ancient one. I am the all-knowing, all-seeing.State your Question, and I shall provide you with the wisdom you seek. But beware, for my answers may be cryptic and profound. What is it that you wish to know?")

with st.chat_message('user'):
    st.text("Oh great and mighty one, I seek your guidance on the mysteries of the universe. Can you enlighten me about how to make mayonaise?")

user_input=st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        st.text("Seriously bro?")
