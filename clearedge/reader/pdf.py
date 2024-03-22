from clearedge.metadata import Metadata
from clearedge.chunk import Chunk
from clearedge.utils.pdf_utils import (
  add_content_to_subheading,
  check_divide,
  clean_blocks,
  contains_block,
  convert_doc_to_image,
  crop_image_by_block,
  extract_text_from_block,
  get_token_list,
  get_text_and_table_blocks,
  get_unique_tables,
  update_subheading,
  should_perform_ocr,
  sort_blocks_by_reading_order
)
from rapidocr_onnxruntime import RapidOCR
from datetime import datetime
from typing import Optional, List
from rapid_table import RapidTable
from PIL import Image

import fitz
import requests
import validators
import io
import cv2
import layoutparser as lp

first_line_end_thesh = 0.8

def process_pdf(
  filepath: Optional[str] = None,
  use_ocr: Optional[bool] = False,
  ocr_provider: Optional[str] = "doctr",
) -> List[Chunk]:
  """
  Processes a file from a given filepath and returns a Chunk object.

  This function is designed to handle both local file paths and URLs as input. It reads the content of the pdf file, processes it according, and encapsulates the processed data into a Chunk object which is then returned.

  Parameters:
  filepath (str): The filepath or URL of the file to be processed. This can be a path to a local file or a URL to a remote file.
  use_ocr (str): If this is set to True, then it will always use ocr method to parse the document. Defaults to False
  ocr_provider (str): Name of the ocr provider to be used if no text found in the pdf. Defaults to 'doctr.' Available values are: 'doctr', 'rapidocr.'

  Returns:
  Chunk: List of Chunk class containing the processed data from the file.

  Raises:
  FileNotFoundError: If the file at the given filepath does not exist or is inaccessible.
  ValueError: If the filepath is invalid or if the file content cannot be processed.

  Example:
  >>> from clearedge.reader.pdf import process_pdf
  >>> chunks = process_pdf(filepath="path/to/local/file.pdf")
  >>> chunks = process_pdf(filepath="http://example.com/remote/file.pdf")
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
  if use_ocr or should_perform_ocr(doc):
    return _process_file_with_ocr(doc)
  else:
    return _parse_with_pymupdf(doc)

def _parse_with_pymupdf(doc):
  page_wise_block_list = []
  block_list = []
  chunks = []
  for page_no, page in enumerate(doc):
    if page_no == 0:
      page_data = page.get_text("dict", flags=fitz.TEXT_INHIBIT_SPACES)
      page_data["blocks"] = [
        block for block in page_data["blocks"] if block["type"] == 0
      ]
      [block.update({"list_item_start": False}) for block in page_data["blocks"]]
      page_data["blocks"].sort(key=lambda page_block: page_block["bbox"][1])
      # initialize empty list
      for block_no, block in enumerate(page_data["blocks"]):
        for line_no, line in enumerate(block["lines"]):
          max_size = 0
          for span in line["spans"]:
            if span["size"] > max_size:
              max_size = span["size"]
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
      metadata = Metadata(
        page_no=page_no + 1,
        bbox=bbox,
        doc_type="pdf",
        created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      )
      chunks.append(Chunk(text=curr_page_content, metadata=metadata))

    for page_wise_blocks in page_wise_block_list:
      block_list.extend(page_wise_blocks)

    if len(block_list) == 0:
      return []
    return chunks

def _process_file_with_ocr(doc):
  chunks = []
  ocr = RapidOCR(
    config_path="clearedge/ocr_config/config.yaml",
    rec_model_path='clearedge/reader/en_PP-OCRv4_rec_infer.onnx'
  )
  model = lp.Detectron2LayoutModel(
    'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
  )
  table_engine = RapidTable()
  # convert doc to images
  images = convert_doc_to_image(doc)
  for page_no, image_path in enumerate(images):
    img = cv2.imread(image_path)
    layout = model.detect(img)
    blocks = get_text_and_table_blocks(layout)
    unique_tables = get_unique_tables(layout)
    sorted_blocks = sort_blocks_by_reading_order(blocks, img)
    subheading_content = {}
    subheading_text = None
    for block in sorted_blocks:
      segment_image = crop_image_by_block(img, block)
      output, _ = ocr(segment_image)
      if block.type == "Table":
        if contains_block(block, unique_tables):
          table_html = table_engine(segment_image, output)[0]
          add_content_to_subheading(subheading_content, subheading_text, table_html)
      else:
        if output:
          final_text = extract_text_from_block(output)
          subheading_text = update_subheading(block, final_text, subheading_text)
          add_content_to_subheading(subheading_content, subheading_text, final_text)
    for key, value in subheading_content.items():
      chunks.append(
        Chunk(
          text=" ".join(value),
          metadata=Metadata(
            sub_heading=key,
            page_no=page_no,
            filename=doc.name,
            doc_type="pdf",
            created_at=datetime.now().strftime("%m/%d/%Y %H:%M"),
          )
        )
      )
  print('chunks ', chunks)
  return chunks
