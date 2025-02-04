


import streamlit as st
import os
from PIL import Image
from io import BytesIO
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from tempfile import NamedTemporaryFile
import traceback  # For better error handling

# Define system prompt and instructions

SYSTEM_PROMPTS="""You are an AI Tutor specializing in generating concise explanations and real-world examples for educational topics. Your role is to analyze provided chapter titles or queries, deliver brief and intuitive explanations for each, offer relevant real-world examples, and generate 20 frequently asked questions (FAQs) with answers for each topic. Present all information in a clear, organized, and student-friendly manner.Pls use Markdown format"""

INSTRUCTIONS ="""
* For each provided chapter title:
    1. **Explanation**:
        - Provide a concise and intuitive explanation of the topic, ensuring clarity and accessibility for students.
    2. **Real-World Examples**:
        - Offer relevant real-world examples that illustrate the topic, enhancing understanding and applicability.
    3. **Frequently Asked Questions (FAQs)**:
        - Generate a list of 20 pertinent FAQs related to the each topic.
        - For Math related topic ensure the queries are more on math problems and solution should be solved step wise .(eg.solve for 3x+5=90 , 1step:add -5 to both sides,2step:3x=90,3step:x=30)
        - Provide clear and accurate answers to each question to facilitate student comprehension and engagement.
*For a query on a topic:
    - Ensure that all step by step explanations are provided.
    - Give more such examples to make the concepts clear
    - Provide a tailor  responses to students in an intuitive way. 
*Present the information in a structured format, using headings, bullet points, and numbering to enhance readability.
* Maintain a supportive and encouraging tone throughout, fostering a positive learning environment."""




# Set API keys from Streamlit secrets (Make sure your secrets.toml is properly configured)
try:
    os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_KEY']
    os.environ['GOOGLE_API_KEY'] = st.secrets['GEMINI_KEY']
except KeyError as e:
    st.error(f"API Key Missing: {e}.  Please set your TAVILY_KEY and GEMINI_KEY in secrets.toml")
    st.stop() # Stop execution if keys are missing


MAX_IMAGE_WIDTH = 300

def resize_image_for_display(image_file):
    """Resize image for display purposes."""
    img = Image.open(image_file)
    aspect_ratio = img.height / img.width
    new_height = int(MAX_IMAGE_WIDTH * aspect_ratio)
    img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@st.cache_resource
def get_agent():
    """Initialize and cache the AI agent."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            system_prompt=SYSTEM_PROMPTS,
            instructions=INSTRUCTIONS,
            tools=[TavilyTools(api_key=os.getenv("TAVILY_API_KEY"))],
            markdown=True,
        )
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        st.error(traceback.format_exc())
        return None


def analyze_image(image_path):
    """Analyze the image to extract chapter names and generate educational content."""
    agent = get_agent()
    if agent is None:
        return #Agent creation failed.

    try:
        with st.spinner('Analyzing image and generating content...'):
            response = agent.run(
                "Analyze the given image to extract chapter names and generate educational content.",
                images=[image_path],
            )
            st.markdown(response.content)
    except Exception as e:
        st.error(f"An error occurred during image analysis: {e}")
        st.error(traceback.format_exc()) # Shows the full traceback for debugging

def save_uploaded_file(uploaded_file):
    """Save the uploaded image to a temporary file."""
    with NamedTemporaryFile(dir='.', suffix='.jpg', delete=False) as f:
        f.write(uploaded_file.getbuffer())
        return f.name

def main():
    st.title("📘 AI-Powered Tutor Agent")
    st.write("Upload an image containing chapter names to receive explanations, real-world examples, and FAQs for each topic.")

    # Improved session state handling - no need for separate if statements.
    if 'agent' not in st.session_state:
        st.session_state.agent = get_agent() # Initialize the agent only once.


    tab1, tab2, tab3 = st.tabs(["📝 Enter Text", "📤 Upload Image", "📸 Take Photo"])

    with tab1:
        user_input = st.text_area("Enter text for analysis:", placeholder="Type or paste the chapter content here...", height=200)
        if st.button("Get Answer"):
            if user_input:
                try:
                    with st.spinner("Searching for info..."):
                        response = st.session_state.agent.run(user_input, markdown=True) #Use cached agent.
                        st.markdown(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error(traceback.format_exc()) #More debugging info
            else:
                st.warning("Please enter a topic.")

    with tab2:
        uploaded_file = st.file_uploader("Upload image of query or chapter names", type=["jpg", "jpeg", "png"], help="Upload a clear image of query or chapter names")
        if uploaded_file:
            resized_image = resize_image_for_display(uploaded_file)
            st.image(resized_image, caption="Uploaded Image", use_container_width=False, width=MAX_IMAGE_WIDTH)
            if st.button("🔍 Analyze Uploaded Image", key="analyze_upload"):
                temp_path = save_uploaded_file(uploaded_file)
                analyze_image(temp_path)
                os.unlink(temp_path)


    with tab3:
        camera_photo = st.camera_input("Take a picture of your query or chapter names")
        if camera_photo:
            resized_image = resize_image_for_display(camera_photo)
            st.image(resized_image, caption="Captured Photo", use_container_width=False, width=MAX_IMAGE_WIDTH)
            if st.button("Analyze Query", key="analyze_camera"):
                temp_path = save_uploaded_file(camera_photo)
                analyze_image(temp_path)
                os.unlink(temp_path)


if __name__ == "__main__":
    st.set_page_config(page_title="AI-Powered Tutor Agent", layout="wide", initial_sidebar_state="collapsed")
    main()
