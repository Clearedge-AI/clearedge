from tqdm import tqdm
from clearedge.metadata import Metadata
from clearedge.chunk import Chunk
from clearedge.utils.pdf_utils import (
  clean_blocks,
  check_divide,
  get_token_list,
)
import fitz
import requests
import validators
import io

first_line_end_thesh = 0.8

def process_file(filepath):
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
  >>> chunk = process_file("path/to/local/file.pdf")
  >>> chunk = process_file("http://example.com/remote/file.pdf")
  """
  if validators.url(filepath):
    response = requests.get(filepath)
    if response.status_code == 200:
      # Create a bytes stream from the response content
      pdf_stream = io.BytesIO(response.content)
      # Use fitz to open the PDF from the bytes stream
      doc = fitz.open("pdf", pdf_stream)
    else:
      print(f"Failed to fetch PDF from {filepath}")
      return []
  else:
    # filepath is not a URL, directly open the PDF from the file path
    try:
      doc = fitz.open(filepath)
    except Exception as e:
      print(f"Failed to open PDF: {e}")
      return []
  return parse_with_pymupdf(doc)

def parse_with_pymupdf(doc):
  page_wise_block_list = []
  block_list = []
  chunks = []
  num_pages = doc.page_count

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
      # for block_lines in block['lines']:
      #   for block_line in block_lines['spans']:

    metadta = Metadata(
      title="",
      section_header="",
      page=page_no + 1,
      bbox=bbox,
    )
    chunks.append(Chunk(text=curr_page_content, metadata=metadta))

  for page_wise_blocks in page_wise_block_list:
    block_list.extend(page_wise_blocks)

  if len(block_list) == 0:
    return []

  return chunks
