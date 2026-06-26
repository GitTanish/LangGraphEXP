import streamlit as st


# session state -> dict -> 
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = [] 


# loop to load conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.text(message["content"])


# {"role":"user", "content":"Hi"}
# {"role":"assistant", "content":"Hello"}

user_input =st.chat_input("Ask me anything...")

if user_input:

    st.session_state["message_history"].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    st.session_state["message_history"].append({'role':'assistant','content':user_input})
    with st.chat_message('assistant'):
        st.text(user_input)
