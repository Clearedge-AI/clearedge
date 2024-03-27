from clearedge.chunk import Chunk
from clearedge.metadata import Metadata
from clearedge.utils.pdf_utils import (
  check_divide,
  clean_blocks,
  convert_to_images,
  get_token_list,
  process_images,
  should_process_with_ocr
)
from rapidocr_onnxruntime import RapidOCR
from typing import Optional, List
from rapid_table import RapidTable
from doctr.models import ocr_predictor
from clearedge.utils.ultralytics_utils import YOLO
from tqdm import tqdm

import fitz
import requests
import validators
import io
import os
import pathlib

first_line_end_thesh = 0.8

def process_pdf(
  filepath: Optional[str] = None,
  use_ocr: Optional[bool] = False,
  ocr_model: Optional[str] = "Tesseract",
) -> List[Chunk]:
  """
  Processes a file from a given filepath and returns a Chunk object.

  This function is designed to handle both local file paths and URLs as input. It reads the content of the pdf file, processes it according, and encapsulates the processed data into a Chunk object which is then returned.

  Parameters:
  filepath (str): The filepath or URL of the file to be processed. This can be a path to a local file or a URL to a remote file.
  use_ocr (bool): Use ocr to parse the pdf file. Defaults to False.
  ocr_model (str): Name of the ocr model to be used if no text found in the pdf. Defaults to 'Tesseract.' Available values are: 'DoctTR', 'Tesseract', 'Rapid'.'

  Returns:
  Chunk: List of Chunk class containing the processed data from the file.

  Raises:
  FileNotFoundError: If the file at the given filepath does not exist or is inaccessible.
  ValueError: If the filepath is invalid or if the file content cannot be processed.

  Example:
  >>> from clearedge.reader.pdf import process_pdf
  >>> chunks = process_pdf(filepath="path/to/local/file.pdf")
  >>> chunks = process_pdf(filepath="http://example.com/remote/file.pdf")
  >>> chunks = process_pdf(filepath="http://example.com/remote/file.pdf", use_ocr=True)
  """
  pdf_stream = None
  doc = None

  # Check if filepath is a URL
  if validators.url(filepath):
    response = requests.get(filepath)
    if response.status_code == 200:
      pdf_stream = io.BytesIO(response.content)
      doc = fitz.open(stream=pdf_stream, filetype="pdf")
    else:
      raise FileNotFoundError(f"Failed to fetch PDF from {filepath}")
  else:
    # Check if the local file exists and is a PDF
    if not filepath.lower().endswith('.pdf'):
      raise ValueError("Filepath does not point to a PDF file.")
    if not os.path.exists(filepath):
      raise FileNotFoundError(f"The file at {filepath} does not exist or is inaccessible.")
    with open(filepath, 'rb') as file:
      pdf_stream = io.BytesIO(file.read())
      doc = fitz.open(filepath)
  if use_ocr or should_process_with_ocr(doc):
    return _process_file_with_ocr(ocr_model, pdf_stream, doc.name)
  else:
    return _fast_process(doc)


def _process_file_with_ocr(ocr_model, pdf_stream, filename):
  current_script_dir = os.path.dirname(__file__)
  # Construct the absolute path to the config.yaml file
  config_path = os.path.join(current_script_dir, '..', 'ocr_config', 'config.yaml')
  config_path = os.path.abspath(config_path)

  # Construct the absolute path to the model file
  rec_model_path = os.path.join(current_script_dir, 'en_PP-OCRv4_rec_infer.onnx')
  rec_model_path = os.path.abspath(rec_model_path)
  images = convert_to_images(pdf_stream)
  # Load the document segmentation model
  docseg_model_name = 'DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet'
  docseg_model = YOLO(docseg_model_name)

  # Process the images with the model
  results = docseg_model(source=images, iou=0.1, conf=0.25)

  # Initialize a dictionary to store results
  mydict = {}
  names = {0: 'Caption', 1: 'Footnote', 2: 'Formula', 3: 'List-item', 4: 'Page-footer', 5: 'Page-header', 6: 'Picture', 7: 'Section-header', 8: 'Table', 9: 'Text', 10: 'Title'}

  # Extract and store the paths, coordinates, and labels of detected components
  for entry in results:
    thepath = pathlib.Path(entry.path)
    thecoords = entry.boxes.xyxyn.cpu().numpy()  # Move tensor to CPU before conversion
    labels = entry.boxes.cls.cpu().numpy()  # Move tensor to CPU before conversion
    num_boxes = len(thecoords)

    # Combine coordinates with corresponding labels
    bbox_labels = []
    for i in range(num_boxes):
      xmin, ymin, xmax, ymax = thecoords[i]
      label = names[int(labels[i])]  # Convert label index to label name using the provided dictionary 'names'
      if label != 'Footnote' and label != 'Page-footer' and label != 'Page-header' and label != 'Picture':
        bbox_labels.append((xmin, ymin, xmax, ymax, label))

    mydict.update({str(thepath): bbox_labels})

  predictor = ocr_predictor(det_arch="fast_tiny", pretrained=True) if ocr_model == "DocTr" else None  # Initialize your OCR predictor here if using DocTr
  rapid_ocr = RapidOCR(config_path=config_path, rec_model_path=rec_model_path)
  table_engine = RapidTable()

  output_data = process_images(images, mydict, ocr_model, filename, predictor, rapid_ocr, table_engine)
  return output_data


def _fast_process(doc):
  page_wise_block_list = []
  block_list = []
  output = []
  num_pages = doc.page_count

  for page_no in tqdm(range(num_pages)):
    page = doc.load_page(page_no)

    page_data = page.get_text("dict", flags=fitz.TEXT_INHIBIT_SPACES)

    page_data["blocks"] = [
      block for block in page_data["blocks"] if block["type"] == 0
    ]
    [block.update({'list_item_start': False}) for block in page_data["blocks"]]

    # initialize empty list
    for block_no, block in enumerate(page_data["blocks"]):
      for line_no, _ in enumerate(block["lines"]):
        page_data["blocks"][block_no]["lines"][line_no]["tokens"] = []
        page_data["blocks"][block_no]["lines"][line_no]["word_bbox"] = []

    # Add word tokens and bbox to lines
    word_data_list = page.get_text("words")

    for word_data in word_data_list:
      block_no = word_data[5]
      line_no = word_data[6]
      bbox = list(word_data[:4])
      bbox[0] = bbox[0] / page_data["width"]
      bbox[1] = bbox[1] / page_data["height"]
      bbox[2] = bbox[2] / page_data["width"]
      bbox[3] = bbox[3] / page_data["height"]
      page_data["blocks"][block_no]["lines"][line_no]["tokens"].append(
        word_data[4]
      )
      page_data["blocks"][block_no]["lines"][line_no]["word_bbox"].append(
        tuple(bbox + [page_no])
      )

    page_data["blocks"] = clean_blocks(page_data["blocks"])
    divided_block_list = []
    for block in page_data["blocks"]:
      divided_block_list.extend(check_divide(block))
    page_data["blocks"] = clean_blocks(divided_block_list)

    page_wise_block_list.append(page_data["blocks"])

  for page_no, blocks in enumerate(page_wise_block_list):
    curr_segment_list = [get_token_list(block) for block in blocks]
    curr_page_content = '\n\n'.join([" ".join(segment["tokens"]) for segment in curr_segment_list])
    bbox = []
    individual_text = []
    for block in blocks:
      x1 = block['bbox'][0]
      y1 = block['bbox'][1]
      x2 = block['bbox'][2]
      y2 = block['bbox'][3]
      bbox.append({"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1})
      individual_text.append(str(block['lines'][0]['spans'][0]['text']))

    meta_data = Metadata(page_no=page_no + 1, bbox=bbox, filename=doc.name, doc_type="pdf")
    output.append(Chunk(text=curr_page_content, metadata=meta_data))

  for page_wise_blocks in page_wise_block_list:
    block_list.extend(page_wise_blocks)

  if len(block_list) == 0:
    return []

  return output
