import streamlit as st
from pathlib import Path
import os
# from RawFileExacter import convert_folder_to_mzml

def convert_folder_to_mzml(input_folder, output_folder, include_ms2, filter_threshold):
    pass



def input_folder_selector(folder_path='.'):
    folders = os.listdir(folder_path)
    selected_folder = st.selectbox('Select a input folder', folders, key='input_folder')
    return os.path.join(folder_path, selected_folder)

def output_folder_selector(folder_path='.'):
    folders = os.listdir(folder_path)
    selected_folder = st.selectbox('Select a out folder', folders, key='output_folder')
    return os.path.join(folder_path, selected_folder)



# Streamlit App
def main():
    st.title("Raw data compressor")

    st.sidebar.header("Input")

    # Select input folder
    input_folder = input_folder_selector()

    st.file_uploader("Upload raw files", type=["raw", "mzML"])

    output_folder = output_folder_selector()

    include_ms2 = st.sidebar.checkbox("Include MS2", value=False)

    filter_threshold = st.sidebar.number_input(
        "Filter threshold", min_value=0, value=None, step=1
    )

    if st.sidebar.button("Start conversion"):
        if not input_folder or not output_folder:
            st.error("Please provide input and output folder paths")
        else:
            with st.spinner("Converting raw files to mzML"):
                result = convert_folder_to_mzml(input_folder, output_folder, include_ms2, filter_threshold)
            st.success("Conversion completed")

if __name__ == "__main__":
    main()