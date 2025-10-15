import nltk
import os

# Define where NLTK data should be downloaded
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")

# Create folder if it doesn't exist
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Add the path to NLTK data
nltk.data.path.append(nltk_data_path)

# Download required resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)

print("NLTK data downloaded successfully!")
