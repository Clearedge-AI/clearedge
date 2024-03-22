from collections import defaultdict
from typing import List
from clearedge.metadata import Metadata
from clearedge.chunk import Chunk
from datetime import datetime
from layoutparser import Layout, Interval, Rectangle

import re

first_line_end_thesh = 0.8

def should_perform_ocr(doc):
  """
  Function to check if text extraction is possible from a PDF file.

  Parameters:
  doc (str): PDF file.

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
          span_hash for span_hash in line1_hash if span_hash in line2_hash]

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

def get_corner_bboxes(bboxes):
  top_left = min(box[0] for box in bboxes)
  top_right = max(box[1] for box in bboxes)
  bottom_right = max(box[2] for box in bboxes)
  bottom_left = min(box[3] for box in bboxes)

  return top_left, top_right, bottom_right, bottom_left

def extract_text(input_data):
  result = []
  current_text = ""
  current_y = None

  for item in input_data:
    _, text, y, _ = item

    if current_y is None or abs(y - current_y) <= 20:
      current_text += " " + text
      current_y = y
    else:
      result.append(current_text.strip())
      current_text = text
      current_y = y

  if current_text:
    result.append(current_text.strip())

  return result

def convert_doc_to_image(doc):
  images = []  # List to store image paths

  # Iterate through each page of the document
  for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    pix = page.get_pixmap()
    image_path = f"page_{page_num}.png"
    pix.save(image_path)
    images.append(image_path)

  return images

def is_intersected(rect1: Rectangle, rect2: Rectangle):
  """Checks if two Rectangles intersect."""
  # Check for horizontal overlap
  if rect1.x_1 < rect2.x_2 and rect2.x_1 < rect1.x_2:
    # Check for vertical overlap
    if rect1.y_1 < rect2.y_2 and rect2.y_1 < rect1.y_2:
      return True
  return False

def filter_unique_non_overlapping_tables(blocks: List[any]):
  """Filters unique table blocks with no overlapping regions."""
  unique_tables = []
  seen_regions = []
  for table_block in blocks:
    table_region = table_block.block
    if any(is_intersected(table_region, seen_region) for seen_region in seen_regions):
      continue  # Skip if overlapping
    unique_tables.append(table_block)
    seen_regions.append(table_region)  # Add the region to the list
  return unique_tables

def contains_block(block, tables):
  for table in tables:
    if table.block.x_1 == block.block.x_1 and table.block.y_1 == block.block.y_1 and table.block.x_2 == block.block.x_2 and table.block.y_2 == block.block.y_2:
      return True
  return False

def get_text_and_table_blocks(layout):
  return Layout([b for b in layout if b.type in ["Text", "Title", "List", "Table", "Figure"]])

def get_unique_tables(layout):
  table_blocks = Layout([b for b in layout if b.type == "Table"])
  return filter_unique_non_overlapping_tables(table_blocks)

def crop_image_by_block(image, block):
  x_1 = int(block.block.x_1)
  y_1 = int(block.block.y_1)
  x_2 = int(block.block.x_2)
  y_2 = int(block.block.y_2)
  return image[y_1:y_2, x_1:x_2]

def extract_text_from_block(output):
  final_text = ""
  for item in output:
    final_text += " " + item[1]
  return final_text

def update_subheading(block, text, subheading_text):
  if block.type == "Title":
    return text
  else:
    return subheading_text

def add_content_to_subheading(content_dict, subheading, text):
  if subheading is not None and subheading != text:
    if subheading in content_dict:
      content_dict[subheading].append(text)
    else:
      content_dict[subheading] = [text]

def sort_blocks_by_reading_order(blocks, image):
  """Sorts TextBlocks in the layout based on their reading order (top-to-bottom, left-to-right)."""

  _, w = image.shape[:2]  # Get image height and width

  # Divide the image into left and right halves
  left_interval = Interval(0, w / 2 * 1.05, axis='x').put_on_canvas(image)
  left_blocks = blocks.filter_by(left_interval, center=True)
  right_blocks = Layout([b for b in blocks if b not in left_blocks])

  # Sort blocks within each half based on their y-coordinate (top-to-bottom)
  left_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)
  right_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

  # Combine the sorted left and right blocks and assign indices
  sorted_blocks = Layout([b.set(id=idx) for idx, b in enumerate(left_blocks + right_blocks)])

  return sorted_blocks
