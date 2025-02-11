import requests
import streamlit as st
import ultralytics
import PIL
from PIL import Image, ImageDraw
from ultralytics import YOLO
import io
import base64

model = YOLO("yolo11n.pt")

class_names = ['boots', 'gloves', 'helmet', 'human', 'vest']

st.title("PPE Detection with YOLO")

image_url = st.text_input("Enter Image URL (Optional)", "")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


def is_base64_image(data):
    return data.startswith("data:image/")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_mode = "Uploaded"
elif image_url:
    if is_base64_image(image_url):
        # Process base64 encoded image
        header, encoded = image_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_mode = "Base64"
    else:
        # Process regular URL
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                image_mode = "URL"
            else:
                st.error(f"Error: Invalid image URL. Status code: {response.status_code}")
                image_mode = None
        except Exception as e:
            st.error(f"Error fetching image from URL: {e}")
            image_mode = None
else:
    st.info("Please upload an image or enter a valid image URL to start detection.")
    image_mode = None

if image_mode is not None:
    st.image(image, caption=f"Image ({image_mode})", use_column_width=True)

    if st.button("Detect PPE"):
        with st.spinner("Detecting PPE..."):
            results = model(image)

            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf
                classes = result.boxes.cls

                draw = ImageDraw.Draw(image)
                for box, conf, cls in zip(boxes.tolist(), confidences.tolist(), classes.tolist()):
                    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                    # Use class name instead of index
                    label = f"{class_names[int(cls)]} {conf:.2f}"
                    draw.text((x1, y1), label, fill="red")

            st.image(image.resize((image.width * 2, image.height * 2)), caption="Detected PPE", use_column_width=True)

st.markdown("---")
st.markdown("## About this App")
st.markdown("This app uses a YOLO model to detect Personal Protective Equipment (PPE) in images.")
