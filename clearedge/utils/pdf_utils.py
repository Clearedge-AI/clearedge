from PIL import Image
from clearedge.bbox import Bbox
from clearedge.chunk import Chunk
from clearedge.metadata import Metadata
from doctr.io import DocumentFile
from pdf2image import convert_from_bytes
from collections import defaultdict

import pytesseract
import numpy as np
import re

first_line_end_thesh = 0.8


def convert_to_images(pdf_stream):
  """Converts a pdf stream into images for ocr processing."""
  images = convert_from_bytes(pdf_stream.read())
  paths = []
  # Save images
  for i, image in enumerate(images):
    image.save(f'page{i}.png', 'PNG')
    paths.append(f'page{i}.png')
  return paths


def process_ocr(ocr_model, image, label, predictor=None, rapid_ocr=None, table_engine=None):
  if ocr_model == 'DocTr':
    cropped_doc = DocumentFile.from_images(image)
    result = predictor(cropped_doc)
    raw_text = [word.value for block in result.pages[0].blocks for line in block.lines for word in line.words]
  elif ocr_model == 'Tesseract':
    data = pytesseract.image_to_string(image, lang='eng')
    raw_text = data.split(" ") if label != "Table" else table_engine(np.array(image), rapid_ocr(np.array(image))[0])[0]
  elif ocr_model == "Rapid":
    out = rapid_ocr(np.array(image))[0]
    raw_text = " ".join(text[1] for text in out) if label != 'Table' else table_engine(np.array(image), out)[0]
  cleaned_text = [text.replace('\x0c', '').replace('\n', ' ') for text in raw_text]
  return " ".join(cleaned_text).rstrip() if label != 'Table' else raw_text


def sort_and_update_metadata(temp_output, width):
  first_third_width = width / 3
  first_group, second_group = [], []

  for bbox_data in temp_output:
    bbox = bbox_data.metadata.bbox
    (first_group if bbox.left < first_third_width else second_group).append(bbox_data)

  sorted_output = sorted(first_group, key=lambda x: x.metadata.bbox.top) + \
      sorted(second_group, key=lambda x: x.metadata.bbox.top)
  return sorted_output


def process_images(img_list, mydict, ocr_model, filename, predictor=None, rapid_ocr=None, table_engine=None):
  output, page_no = [], 0
  last_title, last_subheading = "", ""

  for img_path in img_list:
    temp_output, page_no = [], page_no + 1
    bbox_labels = mydict[img_path]
    img = Image.open(img_path)
    width, height = img.size
    for bbox in bbox_labels:
      xmin, ymin, xmax, ymax, label = bbox
      xmin, ymin, xmax, ymax = [int(coord * size) for coord, size in zip(bbox[:4], [width, height, width, height])]
      cropped_img = img.crop((xmin, ymin, xmax, ymax))
      raw_text = process_ocr(ocr_model, cropped_img, label, predictor, rapid_ocr, table_engine)
      if label in ['Title', 'Section-header']:
        last_title, last_subheading = (raw_text, last_subheading) if label == 'Title' else (last_title, raw_text)

      metadata = Metadata(
        chunk_type=label,
        title=last_title,
        sub_heading=last_subheading,
        page_no=page_no,
        bbox=Bbox(top=ymin, left=xmin, width=xmax - xmin, height=ymax - ymin),
        doc_type="pdf",
        filename=filename
      )
      temp_output.append(Chunk(text=raw_text, metadata=metadata))
    sorted_output = sort_and_update_metadata(temp_output, width)
    for bbox_data in sorted_output:
      metadata = bbox_data.metadata
      if metadata.chunk_type == "Title":
        last_title = bbox_data.text
      elif metadata.chunk_type == "Section-header":
        last_subheading = bbox_data.text
      bbox_data.metadata.title = last_title
      bbox_data.metadata.sub_heading = last_subheading
      output.append(bbox_data)
  return output


def detect_list_item(line):
  initial_token = line["tokens"][0]
  return re.match(
    r"^\s*\(?([a-z]|[A-Z]|\d{1,3}|(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})|(xc|xl|l?x{0,3})(ix|iv|b?i{0,3}))(\.|\)){1,2}\s*$",
    initial_token,
  )


def remove_list_items(block):
  if not block["lines"] or not block["lines"][0]["tokens"]:
    return block

  if detect_list_item(block["lines"][0]):
    block["list_item_start"] = True
    block["lines"][0]["tokens"] = block["lines"][0]["tokens"][1:]
    block["lines"][0]["word_bbox"] = block["lines"][0]["word_bbox"][1:]
  return block


def remove_empty_lines(block):
  block["lines"] = [line for line in block["lines"] if line["tokens"]]
  return block


def clean_blocks(blocks):
  blocks = [remove_list_items(block) for block in blocks]
  blocks = [remove_empty_lines(block) for block in blocks]
  blocks = [block for block in blocks if block["lines"]]

  return blocks


def get_token_list(block):
  tokens = []
  word_bbox = []
  for line in block["lines"]:
    tokens.extend(line["tokens"])
    word_bbox.extend(line["word_bbox"])
  return {"tokens": tokens, "word_bbox": word_bbox}


def font_hash(span):
  return f'{span["font"]}|{round(span["size"], 1)}|{span["flags"]}|{span["color"]}'


def coherent_check(line1, line2, strict=False):
  if line2['spans'][0]['text'][0].lower() != line2['spans'][0]['text'][0] and line1['spans'][-1]['text'][-1] in ['.', '?', '!']:
    return False
  if strict and (
    line2['spans'][0]['text'][0].lower() !=
    line2['spans'][0]['text'][0] or line1['spans'][-1]['text'][-1] in ['.', '?', '!']
  ):
    return False
  return True


def should_divide(line1, line2, block, line_gaps):
  total_width = block["bbox"][2] - block["bbox"][0]
  gap = round(line2["bbox"][1] - line1["bbox"][3], 1)
  line1_hash = [font_hash(span) for span in line1["spans"]]
  line2_hash = [font_hash(span) for span in line2["spans"]]
  common_hash = [span_hash for span_hash in line1_hash if span_hash in line2_hash]

  if not common_hash:
    return True

  for span_hash in common_hash:
    if span_hash in line_gaps and len(line_gaps[span_hash].keys()) > 1:
      all_gaps = sorted(line_gaps[span_hash].keys())
      if gap != all_gaps[0] and gap >= 2 * all_gaps[0]:
        return True

  # detect bullet points
  if detect_list_item(line2) and line1['bbox'][0] > line2['bbox'][0]:
    return True

  if coherent_check(line1, line2):
    return False
  if (line1["bbox"][2] - block["bbox"][0]) / total_width > first_line_end_thesh:
    return False
  if (line2["bbox"][2] - block["bbox"][0]) / total_width < first_line_end_thesh:
    return False
  return True


def check_divide(block):
  blocks = []
  lines = []

  line_gaps = {}

  for i, line in enumerate(block["lines"][:-1]):
    gap = round(block["lines"][i + 1]["bbox"][1] - line["bbox"][3], 1)

    if gap > 0:
      line1_hash = [font_hash(span) for span in line["spans"]]
      line2_hash = [font_hash(span) for span in block["lines"][i + 1]["spans"]]
      common_hash = [
        span_hash for span_hash in line1_hash if span_hash in line2_hash
      ]

      for span_hash in common_hash:
        line_gaps[span_hash] = line_gaps[span_hash] if span_hash in line_gaps else defaultdict(
            int)
        line_gaps[span_hash][gap] += 1

  for i, line in enumerate(block["lines"]):
    lines.append(line)
    if i < len(block["lines"]) - 1 and should_divide(line, block["lines"][i + 1], block, line_gaps):
      blocks.append(block.copy())
      blocks[-1]["lines"] = lines
      lines = []

  if lines:
    blocks.append(block.copy())
    blocks[-1]["lines"] = lines

  for i, block in enumerate(blocks):
    if i > 0:
      blocks[i]['list_item_start'] = False
  return blocks


def should_process_with_ocr(doc):
  """
  Function to check if text extraction is possible from a PDF file.

  Parameters:
  filepath (str): The path to the PDF file.

  Returns:
  bool: True if text extraction is possible, False otherwise.
  """
  try:
    num_pages = doc.page_count

    # Limit the number of pages to check to the first 5 pages or total number of pages, whichever is smaller
    count = min(num_pages, 5)

    content_counter = 0

    # Iterate through the pages
    for idx in range(count):
      page = doc.load_page(idx)

      # Check if the page has text content
      if page.get_text():
        content_counter += 1

    print(f"Checking if text is extractable from PDF: {content_counter} pages contain text.")

    # If at least one page contains text, return True
    return content_counter == 0

  except Exception as e:
    # Handle any exceptions and return False
    print(f"Error occurred while checking PDF: {e}")
    return False
