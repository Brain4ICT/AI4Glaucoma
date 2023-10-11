from ultralytics import YOLO
import streamlit as st
import cv2
import config
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

    

    
def _display_detected_frames(conf, model, st_frame, image):
   
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def load_model(model_path):
 
    model_path = "best.pt"
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
 
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")

                        st.write(ex)


def infer_uploaded_video(conf, model):
    diameters = []
    frame_number = 0
    framerate = 0
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )
   
    my_model = YOLO('best.pt')
    if source_video:
        st.video(source_video)
   
    all_bboxes2 = []
    all_frames = []
    
    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    
                    
                    
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            
                                _display_detected_frames(conf,
                                                        model,
                                                        st_frame,
                                                        image
                                                        )
                                results = list(my_model(image, conf=0.128))
                                if results and len(results[0].boxes) > 0:
                                
                                    result = results[0]
                                    boxes = result.boxes
                                        
                                    bbox = boxes.xyxy.tolist()[0]
                                    print(bbox)
                                    all_bboxes2.append(bbox)
                                    print(all_bboxes2)
                                    diameter = np.sqrt(bbox[-2]**2 + bbox[-1]**2)
                                            
                                    print(diameter)
                                    diameters.append(diameter)
                                    frame_number += 1
                                all_frames.append(image) 
                                if frame_number > 0:  # Check if any frames were processed
                                    framerate = vid_cap.get(cv2.CAP_PROP_FPS)
                                
                                    
                            
                                    
   
                               
                                    
                                    
                                
                                    
                        else:
                                vid_cap.release()
                                break
                    total_time_seconds = frame_number /framerate
                                
                    plt.figure(figsize=(10, 6))
                    plt.plot(np.linspace(0, total_time_seconds, num=len(diameters)), diameters, marker='o')
                    plt.xlabel('Time (seconds)')
                    plt.ylabel('Diameter')
                    plt.title('Variation of Diameters over Time')
                                
                    st.pyplot(plt)
                    
                
