from tqdm import tqdm
from collections import defaultdict

from clearedge.utils.pdf_utils import (
  check_merge,
  clean_blocks,
  check_divide,
  get_token_list,
  merge
)
import fitz
import argparse

first_line_end_thesh = 0.8

def process_file(filepath):
  doc = fitz.open(filepath)
  return fast_pdf_parser(doc)

def fast_pdf_parser(doc):
  page_wise_block_list = []
  block_list = []
  line_gaps = {}
  output_json = []
  num_pages = doc.page_count

  for page_no in tqdm(range(num_pages), desc=f"Extracting {doc.name}"):
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
    meta_data = {"title": "", "section_header": "", "page": page_no + 1, "bbox": bbox, "individual_text": individual_text}
    output_json.append({"raw_text": curr_page_content, "metadata": meta_data})

  for page_wise_blocks in page_wise_block_list:
    block_list.extend(page_wise_blocks)

  if len(block_list) == 0:
    return []

  block_list = [{**block, 'idx': idx} for idx, block in enumerate(block_list)]

  redemptions = 2
  redemption_thresh = 3

  for _ in range(redemptions):
    global block_miss
    global block_double

    block_miss = defaultdict(list)
    block_double = defaultdict(list)

    merged_block_list = [block_list[0]]

    for block in block_list[1:]:
      can_merge = check_merge(merged_block_list[-1], block, line_gaps)
      if can_merge:
        merged_block_list[-1] = merge(merged_block_list[-1], block)
      else:
        merged_block_list.append(block)

    block_list = merged_block_list

    for span_hash_gap in block_double.keys():
      if len(block_double[span_hash_gap]) > redemption_thresh:
        span_hash, gap = span_hash_gap.split('~')
        line_gaps[span_hash] = line_gaps[span_hash] if span_hash in line_gaps else []
        line_gaps[span_hash].append(float(gap))

  segment_list = [get_token_list(block) for block in block_list]

  return '\n\n'.join([" ".join(segment["tokens"]) for segment in segment_list]), output_json


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--file_path", help="File path", required=True)
  args = parser.parse_args()
  path = args.file_path
  print(process_file(path))
