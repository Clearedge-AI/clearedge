from clearedge.metadata import Metadata
from clearedge.chunk import Chunk
from clearedge.utils.pdf_utils import (
  clean_blocks,
  check_divide,
  get_token_list,
)
from rapidocr_onnxruntime import RapidOCR
from typing import Optional

import fitz
import requests
import validators
import io
import cv2

first_line_end_thesh = 0.8

class ProcessPDF:
  def __init__(self):
    pass

  def __call__(self, filepath: Optional[str] = None):
    """
    Processes a file from a given filepath and returns a Chunk object.

    This function is designed to handle both local file paths and URLs as input. It reads the content of the pdf file, processes it according, and encapsulates the processed data into a Chunk object which is then returned.

    Parameters:
    filepath (str): The filepath or URL of the file to be processed. This can be a path to a local file or a URL to a remote file.

    Returns:
    Chunk: An instance of the Chunk class containing the processed data from the file.

    Raises:
    FileNotFoundError: If the file at the given filepath does not exist or is inaccessible.
    ValueError: If the filepath is invalid or if the file content cannot be processed.

    Example:
    >>> processor = ProcessPDF()
    >>> chunk = processor("path/to/local/file.pdf")
    >>> chunk = processor("http://example.com/remote/file.pdf")
    """
    if validators.url(filepath):
      response = requests.get(filepath)
      if response.status_code == 200:
        pdf_stream = io.BytesIO(response.content)
        try:
          doc = fitz.open("pdf", pdf_stream)
        except Exception as e:
          raise ValueError(f"Failed to open PDF: {e}")
      else:
        raise FileNotFoundError(f"Failed to fetch PDF from {filepath}")
    else:
      if not filepath.lower().endswith('.pdf'):
        raise ValueError("Filepath does not point to a PDF file.")
      try:
        doc = fitz.open(filepath)
      except Exception as e:
        if "no such file" in str(e).lower():
          raise FileNotFoundError(f"The file at {filepath} does not exist or is inaccessible.")
        else:
          raise ValueError(f"Failed to open PDF: {e}")
    return self.parse_with_pymupdf(doc)

  def parse_with_pymupdf(self, doc):
    page_wise_block_list = []
    block_list = []
    chunks = []
    for page_no, page in enumerate(doc):
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

      for block in blocks:
        x1 = block['bbox'][0]
        y1 = block['bbox'][1]
        x2 = block['bbox'][2]
        y2 = block['bbox'][3]
        bbox.append({"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1})

      metadta = Metadata(
        page_no=page_no + 1,
        bbox=bbox,
      )
      chunks.append(Chunk(text=curr_page_content, metadata=metadta))

    for page_wise_blocks in page_wise_block_list:
      block_list.extend(page_wise_blocks)

    if len(block_list) == 0:
      return []

    return chunks

  def convert_doc_to_image(self, doc):
    return ""

  def process_file_with_ocr(self, doc):
    # convert doc to images
    rapid_ocr = RapidOCR()
    images = self.convert_doc_to_image(doc)
    full_text = ""
    for image_path in images:
      img = cv2.imread(image_path)
      ocr_result, _ = rapid_ocr(img)
      text = ocr_result[1]
      full_text += text
    return full_text
