import zipfile
import os
import subprocess


def unzip_file(zip_file_path, extracted_folder_path):
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract each file one by one
        for file_info in zip_ref.infolist():
            zip_ref.extract(file_info, extracted_folder_path)


def download_data(force=False):
    # Check if the folder exists, and create it if not
    curdir=os.getcwd()
    base_folder = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(base_folder,'data')):
        os.makedirs(os.path.join(base_folder,'data'))
        print(f"Folder '{'data'}' created.")
        # Change to the specified folder
    os.chdir(os.path.join(base_folder,'data'))
    subprocess.check_call(["zenodo_get", 'https://zenodo.org/records/12545278'])
    # Open the zip file
    if force or not os.path.exists('data'):
        # Unzip the file
        unzip_file('subject01.zip', '.')
    os.chdir(curdir)