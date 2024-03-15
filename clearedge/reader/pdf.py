from tqdm import tqdm
from collections import defaultdict
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from io import StringIO
from clearedge.utils.pdf_utils import (
  check_pdf,
  clean_blocks,
  check_divide,
  get_token_list,
)
import fitz
import json
import os
import glob
import argparse
import pandas as pd
# import tensorflow as tf
import numpy as np
import cv2
import layoutparser as lp

first_line_end_thesh = 0.8

def process_file(filepath):
  doc = fitz.open(filepath)
  return fast_pdf_parser(doc)

def fast_pdf_parser(doc):
  page_wise_block_list = []
  block_list = []
  output_json = []
  num_pages = doc.page_count
  print("The number of pages are ", num_pages)

  for page_no in tqdm(range(num_pages), desc=f"Extracting {os.path.basename(doc.name)}"):
      page = doc.load_page(page_no)

      page_data = page.get_text("dict", flags=fitz.TEXT_INHIBIT_SPACES)

      page_data["blocks"] = [
          block for block in page_data["blocks"] if block["type"] == 0]
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
              word_data[4])
          page_data["blocks"][block_no]["lines"][line_no]["word_bbox"].append(
              tuple(bbox + [page_no]))
          

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
          bbox.append({"top": y1, "left": x1, "width": x2-x1, "height": y2-y1})
          individual_text.append(str(block['lines'][0]['spans'][0]['text']))
      meta_data = {"title": "", "section_header": "", "page": page_no+1, "bbox": bbox, "individual_text": individual_text}
      output_json.append({"raw_text": curr_page_content, "metadata": meta_data})

  for page_wise_blocks in page_wise_block_list:
      block_list.extend(page_wise_blocks)
      
  if len(block_list) == 0:
      return []

  # # Saving the text content
  # os.makedirs("./test_output", exist_ok=True)
  # with open(
  #     os.path.join(
  #         "./test_output", ".".join(os.path.basename(filepath).split(".")[:-1])+".txt"),
  #     "w+",
  # ) as f:
  #     segment_list = [get_token_list(block) for block in block_list]

  #     print('\n\n'.join([" ".join(segment["tokens"])
  #             for segment in segment_list]), file=f)

  # segment_list = [get_token_list(block) for block in block_list]

  # # Saving the JSON output
  # # Creating the directory if it doesn't exist
  # os.makedirs("./test_output", exist_ok=True)

  # # Filepath of the JSON file you want to write
  # filepath = "./test_output/{}.json".format(os.path.basename(filepath).split(".")[0])

  # # Writing JSON data to the file
  # with open(filepath, "w+") as f:
  #     json.dump(output_json, f, indent=4)

  segment_list = [get_token_list(block) for block in block_list]
            
  return '\n\n'.join([" ".join(segment["tokens"]) for segment in segment_list]), output_json


def pdf2img(pdf_path):
    # Remove existing JPEG files in the pages directory
  for file in os.listdir('pages/'):
    if file.endswith('.jpg'):
      os.remove(os.path.join('pages/', file))

  # Convert PDF to images
  images = convert_from_path(pdf_path)

  # Save images to the pages directory
  for i, image in enumerate(images):
    image.save(f'pages/page{i}.jpg', 'JPEG')

  return images

def scannedPageOCR(pdf_path):

  # Check if directories 'pages' and 'temp' exist, if not, create them
  if not os.path.exists('pages'):
    os.makedirs('pages')
  if not os.path.exists('temp'):
    os.makedirs('temp')

  images = pdf2img(pdf_path)
  text_content = ""

  # looping over the images stored in pages directory
  for img in range(len(images)):
    image_path = f"pages/page{img}.jpg"
    image = cv2.imread(image_path)
    image = image[..., ::-1]
    model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                                          threshold=0.5,
                                          label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                                          enforce_cpu=True,
                                          enable_mkldnn=True)  # math kernel library
    # Text Extraction
    layout = model.detect(image)
    table_count = 0  # to maintain the count of table on individual page
    text_blocks = lp.Layout([b for b in layout if b.type == 'Text' or b.type == 'Title' or b.type == 'List'])
    h, w = image.shape[:2]

    left_interval = lp.Interval(0, w / 2 * 1.05, axis='x').put_on_canvas(image)

    left_blocks = text_blocks.filter_by(left_interval, center=True)
    left_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)
    # The b.coordinates[1] corresponds to the y coordinate of the region
    # sort based on that can simulate the top-to-bottom reading order
    right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
    right_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

    # And finally combine the two lists and add the index
    text_blocks = lp.Layout([b.set(id=idx) for idx, b in enumerate(left_blocks + right_blocks)])

    ocr_agent = lp.TesseractAgent(languages='eng')
    for block in text_blocks:
      segment_image = (block
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(image))
      # add padding in each image segment can help
      # improve robustness

      text = ocr_agent.detect(segment_image)
      block.set(text=text, inplace=True)
    for txt in text_blocks.get_texts():
      text_content += txt[:-1]

    # Table Extraction
    extracted_tables = set()
    all_tables_df_4 = pd.DataFrame()
    table_count = 1  # to maintain the count of table on individual page
    for l in layout:
      if l.type == 'Table' or l.type == 'Figure':  # in some pdfs tables might be detectable as figure
        x_1 = int(l.block.x_1)
        y_1 = int(l.block.y_1)
        x_2 = int(l.block.x_2)
        y_2 = int(l.block.y_2)
        im = cv2.imread(f"pages/page{img}.jpg")
        if im is not None:
          cv2.imwrite(f'temp/ext_table_{img}_{table_count}.jpg', im[y_1:y_2, x_1:x_2])
          ocr = PaddleOCR(use_gpu=False, lang='en')
          image_path = f'temp/ext_table_{img}_{table_count}.jpg'
          image_cv = cv2.imread(image_path)
          if image_cv is not None:
            image_height = image_cv.shape[0]
            image_width = image_cv.shape[1]
            output = ocr.ocr(image_path)[0]
            if output is not None:
              boxes = [line[0] for line in output]
              texts = [line[1][0] for line in output]
              probabilities = [line[1][1] for line in output]
              horiz_boxes = []
              vert_boxes = []

              for box in boxes:
                x_h, x_v = 0, int(box[0][0])
                y_h, y_v = int(box[0][1]), 0
                width_h, width_v = image_width, int(box[2][0] - box[0][0])
                height_h, height_v = int(box[2][1] - box[0][1]), image_height

                horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
                vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])

              # horiz_out = tf.image.non_max_suppression(
              #     horiz_boxes,
              #     probabilities,
              #     max_output_size=1000,
              #     iou_threshold=0.1,
              #     score_threshold=float('-inf'),
              #     name=None
              # )
              # horiz_lines = np.sort(np.array(horiz_out))

              # vert_out = tf.image.non_max_suppression(
              #     vert_boxes,
              #     probabilities,
              #     max_output_size=1000,
              #     iou_threshold=0.1,
              #     score_threshold=float('-inf'),
              #     name=None
              # )
              # vert_lines = np.sort(np.array(vert_out))

              # out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
              # unordered_boxes = []

              # for i in vert_lines:
              #   unordered_boxes.append(vert_boxes[i][0])
              # ordered_boxes = np.argsort(unordered_boxes)

              def intersection(box_1, box_2):
                return [box_2[0], box_1[1], box_2[2], box_1[3]]

              def iou(box_1, box_2):
                x_1 = max(box_1[0], box_2[0])
                y_1 = max(box_1[1], box_2[1])
                x_2 = min(box_1[2], box_2[2])
                y_2 = min(box_1[3], box_2[3])
                inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
                if inter == 0:
                  return 0
                box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
                box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))
                return inter / float(box_1_area + box_2_area - inter)

              # for i in range(len(horiz_lines)):
              #   for j in range(len(vert_lines)):
              #     resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])
              #     for b in range(len(boxes)):
              #       the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
              #       if (iou(resultant, the_box) > 0.1):
              #         out_array[i][j] = texts[b]

              # out_array = np.array(out_array)
              # table = tuple(map(tuple, out_array))  # Convert to hashable tuple
              # if table not in extracted_tables:  # Check if table is unique
              #   extracted_tables.add(table)
              #   page_df = pd.DataFrame(out_array)
              #   all_tables_df_4 = pd.concat([all_tables_df_4, page_df], ignore_index=True)
              #   table_count += 1

    text_content += "\n"
    output_string = StringIO()
    # all_tables_df_4.to_csv(output_string, sep='\t', index=False)
    formatted_dataframe_string = output_string.getvalue()
    text_content += formatted_dataframe_string

  # Store text_content
  os.makedirs("./test_output", exist_ok=True)
  with open(
      os.path.join("./test_output", os.path.basename(pdf_path).split(".")[0] + ".txt"),
      "w+",
  ) as f:
    f.write(text_content)

  # Store text_content
  os.makedirs("./test_output", exist_ok=True)
  with open(
      os.path.join("./test_output", os.path.basename(pdf_path).split(".")[0] + "test.txt"),
      "w+",
  ) as f:
    f.write(text_content)

  return text_content

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--file_path", help="File path", required=True)
  args = parser.parse_args()
  path = args.file_path
  print(process_file(path))
