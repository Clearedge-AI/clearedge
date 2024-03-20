from collections import defaultdict
from clearedge.metadata import Metadata
from clearedge.chunk import Chunk
from datetime import datetime
import openai
import re

first_line_end_thesh = 0.8

def check_pdf(doc):
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
    return content_counter > 0

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

def get_extreme_bboxes(bboxes):
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

def group_rapidocr_texts_by_bbox(text_items, page_no, use_llm=False, llm_provider="openai"):
  y_threshold = 20
  x_threshold = 4
  # Sort the bounding box data by the y-coordinate of the top-left corner
  text_items.sort(key=lambda x: x[0][0][1])
  current_group = []
  chunks = []
  prev_y = None
  for bbox, text, _ in text_items:
    x, y = bbox[0]
    if prev_y is None or y - prev_y <= y_threshold:
      current_group.append((x, text, y, bbox))
    else:
      # Sort the current group by x-coordinate of the top-left corner and join the texts
      current_group.sort(key=lambda x: x[0])
      current_line = [current_group[0]]
      prev_x = current_group[0][0]

      for i in range(1, len(current_group)):
        if current_group[i][0] - prev_x <= x_threshold:
          current_line.append(current_group[i])
        else:
          # sort by y-coordinate of the top-left corner
          sorted_data = sorted(current_line, key=lambda x: x[2])
          print('sorted_data ', ' '.join(x[1] for x in sorted_data))
          extracted = extract_text(sorted_data)
          boxes = [bbox[3] for bbox in sorted_data]
          top_left, top_right, bottom_right, bottom_left = get_extreme_bboxes(boxes)
          for chunk in extracted:
            chunks.append(
              Chunk(
                text=chunk,
                metadata=Metadata(
                  page_no=page_no,
                  bbox=[top_left, top_right, bottom_right, bottom_left],
                  doc_type="pdf",
                  created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
              )
            )
          current_line = [current_group[i]]
        prev_x = current_group[i][0]
      sorted_data = sorted(current_line, key=lambda x: x[2])
      print('sorted_data ', ' '.join(x[1] for x in sorted_data))

      extracted = extract_text(sorted_data)
      boxes = [bbox[3] for bbox in sorted_data]
      top_left, top_right, bottom_right, bottom_left = get_extreme_bboxes(boxes)
      for chunk in extracted:
        chunks.append(
          Chunk(
            text=chunk,
            metadata=Metadata(
              page_no=page_no,
              bbox=[top_left, top_right, bottom_right, bottom_left],
              doc_type="pdf",
              created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
          )
        )
      current_group = [(x, text, y, bbox)]
    prev_y = max(y, prev_y) if prev_y is not None else y
  # Process the last group
  if current_group:
    current_group.sort(key=lambda x: x[0])
    current_line = [current_group[0]]
    prev_x = current_group[0][0]

    for i in range(1, len(current_group)):
      if current_group[i][0] - prev_x <= x_threshold:
        current_line.append(current_group[i])
      else:
        sorted_data = sorted(current_line, key=lambda x: x[2])
        print('sorted_data ', ' '.join(x[1] for x in sorted_data))
        extracted = extract_text(sorted_data)
        boxes = [bbox[3] for bbox in sorted_data]
        top_left, top_right, bottom_right, bottom_left = get_extreme_bboxes(boxes)
        for chunk in extracted:
          chunks.append(
            Chunk(
              text=chunk,
              metadata=Metadata(

                page_no=page_no,
                bbox=[top_left, top_right, bottom_right, bottom_left],
                doc_type="pdf",
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
              )
            )
          )
        current_line = [current_group[i]]
      prev_x = current_group[i][0]

    sorted_data = sorted(current_line, key=lambda x: x[2])
    print('sorted_data ', ' '.join(x[1] for x in sorted_data))
    extracted = extract_text(sorted_data)
    boxes = [bbox[3] for bbox in sorted_data]
    top_left, top_right, bottom_right, bottom_left = get_extreme_bboxes(boxes)
    for chunk in extracted:
      chunks.append(
        Chunk(
          text=chunk,
          metadata=Metadata(
            page_no=page_no,
            bbox=[top_left, top_right, bottom_right, bottom_left],
            doc_type="pdf",
            created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
          )
        )
      )
  return chunks

def convert_doc_to_image(doc):
  images = []  # List to store image paths

  # Iterate through each page of the document
  for page_num in range(len(doc)):
    page = doc.load_page(page_num)  # Load the current page
    pix = page.get_pixmap()  # Render page to an image
    image_path = f"page_{page_num}.png"  # Define image path
    pix.save(image_path)  # Save the image to disk
    images.append(image_path)  # Append the image path to the list

  return images
