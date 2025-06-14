import streamlit as st

def show_sidebar(mode):
    st.sidebar.title("🖥️ How It Works?")
    
    if mode == "Canvas Input":
        st.sidebar.write("""
        ✍️ **Canvas Drawing Mode**  
        1️⃣ **Draw** a number with multiple digits on the canvas  
        2️⃣ Click **Predict** to recognize the digits  
        3️⃣ View predicted number and confidence graph  
        4️⃣ Download your drawing if needed  
        """)
    
    elif mode == "Camera Input":
        st.sidebar.write("""
        📸 **Camera Input Mode**  
        1️⃣ Use your webcam to capture handwritten digits  
        2️⃣ Click **Predict** to recognize the digits  
        3️⃣ View the results and confidence  
        """)
    
    elif mode == "By uploading":
        st.sidebar.write("""
        📂 **Upload Mode**  
        1️⃣ Upload an image with multiple handwritten digits  
        2️⃣ Click **Predict** to process and recognize  
        3️⃣ View the full predicted number and confidence  
        """)
    
    st.sidebar.title("About Developer 📢")
    st.sidebar.write("""
    👩‍💻 **Sahil Bargotra and Aminder Singh**   
    """)
