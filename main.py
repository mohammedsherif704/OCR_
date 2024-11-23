import os
import sys
import shutil
import argparse
import cv2
from pdf2image import convert_from_path, pdfinfo_from_path
from spire.presentation import *
from spire.presentation.common import *
#from dotenv import load_dotenv
#load_dotenv()
from Component import Layout
#from TextParse.Texter import SeekText



def parse_args(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="OCR service")
    add_arg = parser.add_argument
    add_arg(
        "--input",
        type=str,
        required=True,
        help="Path to the directory containing files to perform OCR",
    )
    add_arg(
        "--output",
        type=str,
        required=True,
        help="Path to save OCR output",
    )
    return parser.parse_args()


def convert_pdf2images(doc_path: str, save_dir: str):
    info = pdfinfo_from_path(doc_path, userpw=None, poppler_path=None)
    maxPages = info["Pages"]
    i = 0
    for page in range(1, maxPages + 1, 10):
        images = convert_from_path(
            doc_path, dpi=200, first_page=page, last_page=min(page + 10 - 1, maxPages)
        )
        for img in images:
            img.save(f"{save_dir}/page_{str(i)}.png")
            component(save_dir, f"page_{str(i)}" , '.png')
            #text_seeker(save_dir, f"page_{str(i)}" , '.png')
            i += 1
    print("Successfully saved", doc_path)


def convert_pptx_to_images(doc_path, save_dir):
    
    presentation = Presentation()
    presentation.LoadFromFile(doc_path)
    presentation.SaveToFile(f"{save_dir}/PresentationToPDF.pdf", FileFormat.PDF)
    presentation.Dispose()

    info = pdfinfo_from_path(f"{save_dir}/PresentationToPDF.pdf")
    maxPages = info["Pages"]
    i = 0
    for page in range(1, maxPages + 1, 10):
        images = convert_from_path(
            f"{save_dir}/PresentationToPDF.pdf",
            dpi=200,
            first_page=page,
            last_page=min(page + 10 - 1, maxPages),
        )
        for img in images:
            img.save(f"{save_dir}/page_{i}.jpg")
            i += 1
    os.remove(f"{save_dir}/PresentationToPDF.pdf")
    print(f"Successfully saved {doc_path}")


def main(data_path, save_path):

    global ocr_obj
    
    for f in os.listdir(data_path):
        fname, ext = os.path.splitext(f)
        if not os.path.exists(f"{save_path}/{fname}"):
            os.makedirs(f"{save_path}/{fname}/images")
        else:
            raise OSError(
                "Directory already exists. Remove the existing directory to proceed."
            )
            sys.exit(1)
        if ext == ".pdf":
            print("Converting File ... Please wait")
            convert_pdf2images(f"{data_path}/{f}", f"{save_path}/{fname}/images")
        elif ext == ".pptx":
            print("Converting File ... Please wait")
            convert_pptx_to_images(f"{data_path}/{f}", f"{save_path}/{fname}/images")
        elif ext == ".png" or ext == ".jpg" or ext == ".jpeg":
            img = cv2.imread(f"{data_path}/{f}")
            cv2.imwrite(f"{save_path}/{fname}/images/page_0.jpg", img)
        else:
            raise ValueError(f"This file format is not yet supported. {f}")
            sys.exit(1)
            

if __name__ == "__main__":
    #args = parse_args()
    global component
    component = Layout()

    #global text_seeker
    #text_seeker = SeekText()
    main(r'data_used', 'output')
