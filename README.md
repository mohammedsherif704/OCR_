# OCR_

!rm -rf output
!mkdir output

!zip -r folder_name.zip folder_name
from google.colab import files
files.download("folder_name.zip")

!sudo apt update
!sudo apt install poppler-utils
!pip install -r requirements.txt
