import streamlit as st
from PIL import Image
import pandas as pd
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from RGB2IMG_indo import *

model_b0 = tf.keras.models.load_model("model.h5")
class_names = ['Gatak', 'Kelihos_ver1', 'Kelihos_ver3', 'Lollipop',
               'Obfuscator.ACY', 'Ramnit', 'Simda', 'Tracur', 'Vundo']
class_asli = ["Ramnit", "Lollipop", "Kelihos_ver3", "Vundo", "Simda",
              "Tracur", "Kelihos_ver1", "Obfuscator.ACY", "Gatak"]


def load_image(image_file):
    img = Image.open(image_file)
    return img


def img_prediction(img_files):
    img = tf.io.read_file(img_files)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img_expand = tf.expand_dims(img, axis=0)
    pred_prob_img = model_b0.predict(img_expand)
    pred_classes_img = class_names[pred_prob_img.argmax()]
    return pred_classes_img, pred_prob_img


def predict_from_bytes(filepath, size=224):
    data_read = theByteFile(filepath)
    data_combine = np.hstack(data_read)

    img_mal = settingArrayImageSize(data_combine, mode="GRAY")
    # im_rgb = settingArrayImageSize(data_combine, mode = "RGB")

    savePictureAsArray(img_mal, "malware_pict.png")
    # savePictureAsArray(im_rgb, "mal_rgb.png")

    # img_prediction("malware_pict.png")
    img = tf.io.read_file("malware_pict.png")
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [size, size])

    img_expand = tf.expand_dims(img, axis=0)

    pred_prob_img = model_b0.predict(img_expand)
    pred_classes_img = class_names[pred_prob_img.argmax()]

    return pred_classes_img, pred_prob_img
    # tf.keras.utils.load_img("malware_pict.png")


# @st.cache()


def main():
    st.title("Malware BIG 2015 Classification")

    menu = ["Image", "Bytes", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Image":
        st.header("Image")
        uploaded_files = st.file_uploader(
            "Upload Images", type=["png", "jpg", "jpeg"],
            accept_multiple_files=True)
        col1, col2, col3 = st.columns(3)
        if uploaded_files is not None:
            for image_file in uploaded_files:
                file_details = {"filename": image_file.name,
                                "filetype": image_file.type,
                                "filesize": image_file.size}
                st.write(file_details)

                with col2:
                    st.image(load_image(image_file), width=250)

                # Saving upload
                with open(os.path.join("img", image_file.name), "wb") as f:
                    f.write((image_file).getbuffer())

                # st.success("File Saved")

                st.header("Prediction")
                predict, prob = img_prediction(f"img/{image_file.name}")
                df_predict = pd.DataFrame(prob, columns=class_names)
                st.bar_chart(df_predict.T)
                st.success(f"The image predicted as {predict}")

    elif choice == "Bytes":
        st.header("Bytes")
        data_file = st.file_uploader("Upload CSV", type=["bytes"])

        if data_file is not None:
            file_details = {"filename": data_file.name,
                            "filetype": data_file.type,
                            "filesize": data_file.size}

            # Saving upload
            with open(os.path.join("bytes", data_file.name), "wb") as f:
                f.write((data_file).getbuffer())

            st.header("Preview File")
            st.write(file_details)
            df = pd.read_csv(data_file)
            st.dataframe(df.head())
            # st.write(data_file.name)

            predict, prob = predict_from_bytes(f"bytes/{data_file.name}")

            st.header("The Visualization")
            col1, col2, col3 = st.columns(3)

            with col2:
                st.image(load_image("malware_pict.png"), width=250)

            st.header("Prediction")
            st.write(predict)

            df_predict = pd.DataFrame(prob, columns=class_names)
            st.bar_chart(df_predict.T)
            st.success(f"The image predicted as {predict}")

    elif choice == "About":
        st.header("About")
        st.markdown('<div style="text-align: justify">This app was created with Streamlit. The model implemented in this application was trained by EfficientNet algorithm to classify 9 classes of malware big 2015. Input from this application can be an image or bytes files from malware big 2015.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:right">~ handhikayp</div>', unsafe_allow_html=True)
        # col1, col2, col3 = st.columns(3)
        # with col2:
            # st.image("img/Handhika.jpg")
            # st.markdown(
            #     '<div style="text-align: center"> Handhika YP </div>', unsafe_allow_html=True)
        # with col3:

        st.header("Check the name of malware family")
        df = pd.read_csv("trainLabels.csv")
        make_choice = st.selectbox("Check the name of malware family", df["Id"])
        mal_class = df["Class"].loc[df["Id"] == make_choice].values[0]
        mal_fams = df["Family"].loc[df["Id"] == make_choice].values[0]

        st.write(f"Malware with name '{make_choice}' is included in {mal_fams} family")


if __name__ == "__main__":
    main()
