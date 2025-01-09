import requests
import streamlit as st
import ultralytics
import PIL
from PIL import Image, ImageDraw
from ultralytics import YOLO
import io
import base64

# Load the YOLO model
model = YOLO("best.pt")  # Ensure 'best.pt' is in the correct directory

# Define class names corresponding to your model
class_names = ['boots', 'gloves', 'helmet', 'human', 'vest']  # Update this list based on your model's classes

st.title("PPE Detection with YOLO")

# User input for image URL and file upload
image_url = st.text_input("Enter Image URL (Optional)", "")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


# Function to check if a string is a base64 image
def is_base64_image(data):
    return data.startswith("data:image/")


# Process uploaded image or image from URL
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

# If an image is available, display it and perform detection
if image_mode is not None:
    st.image(image, caption=f"Image ({image_mode})", use_column_width=True)

    if st.button("Detect PPE"):
        with st.spinner("Detecting PPE..."):
            # Run inference on the input image
            results = model(image)  # Directly pass the PIL Image

            # Process each result in the results list
            for result in results:
                # Get bounding boxes and labels
                boxes = result.boxes.xyxy  # Get bounding box coordinates (x1, y1, x2, y2)
                confidences = result.boxes.conf  # Get confidence scores
                classes = result.boxes.cls  # Get class indices

                # Draw bounding boxes on the original image
                draw = ImageDraw.Draw(image)
                for box, conf, cls in zip(boxes.tolist(), confidences.tolist(), classes.tolist()):
                    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)  # Draw rectangle

                    # Use class name instead of index
                    label = f"{class_names[int(cls)]} {conf:.2f}"  # Get class name and confidence
                    draw.text((x1, y1), label, fill="red")  # Draw label with confidence

            # Show the processed image with detections at high resolution
            st.image(image.resize((image.width * 2, image.height * 2)), caption="Detected PPE", use_column_width=True)

            # Display results as a DataFrame (optional)
            # df = results.pandas().xyxy[0]
            # if not df.empty:
            #     st.write("Detection Results:")
            #     st.dataframe(df)
            # else:
            #     st.write("No PPE detected in the image.")

st.markdown("---")
st.markdown("## About this App")
st.markdown("This app uses a YOLO model to detect Personal Protective Equipment (PPE) in images.")
