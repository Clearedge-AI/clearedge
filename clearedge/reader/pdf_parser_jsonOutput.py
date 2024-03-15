# Install PyMuPDF Module  =>  !pip install PyMuPDF
# !pip install pdf2image
# !sudo apt-get install poppler-utils
# !python3 -m pip install paddlepaddle-gpu
# !pip install "paddleocr>=2.0.1"
# !pip install protobuf==3.20.0
# !git clone https://github.com/PaddlePaddle/PaddleOCR.git
# !wget https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
# !pip install -U layoutparser-0.0.0-py3-none-any.whl
# !python -m pip install paddlepaddle
import fitz
from tqdm import tqdm
import json
import os
import re
import glob
import argparse
from collections import defaultdict
import re
import pprint
from pdf2image import convert_from_path
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import layoutparser as lp
from paddleocr import PaddleOCR, draw_ocr
from io import StringIO

first_line_end_thesh = 0.8


def check_pdf(filepath):
    """
    Function to check if text extraction is possible from a PDF file.

    Parameters:
    filepath (str): The path to the PDF file.

    Returns:
    bool: True if text extraction is possible, False otherwise.
    """
    try:
        doc = fitz.open(filepath)
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
            line2['spans'][0]['text'][0] or line1['spans'][-1]['text'][-1] in ['.', '?', '!']):
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
        gap = round(block["lines"][i+1]["bbox"][1] - line["bbox"][3], 1)

        if gap > 0:
            line1_hash = [font_hash(span) for span in line["spans"]]
            line2_hash = [font_hash(span) for span in block["lines"][i+1]["spans"]]
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


def pdf_parser(filepath):
    doc = fitz.open(filepath)

    page_wise_block_list = []
    block_list = []
    output_json = []
    num_pages = doc.page_count
    print("The number of pages are ", num_pages)

    for page_no in tqdm(range(num_pages), desc=f"Extracting {os.path.basename(filepath)}"):
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

    # Saving the text content
    os.makedirs("./test_output", exist_ok=True)
    with open(
        os.path.join(
            "./test_output", ".".join(os.path.basename(filepath).split(".")[:-1])+".txt"),
        "w+",
    ) as f:
        segment_list = [get_token_list(block) for block in block_list]

        print('\n\n'.join([" ".join(segment["tokens"])
                for segment in segment_list]), file=f)

    segment_list = [get_token_list(block) for block in block_list]

    # Saving the JSON output
    # Creating the directory if it doesn't exist
    os.makedirs("./test_output", exist_ok=True)

    # Filepath of the JSON file you want to write
    filepath = "./test_output/{}.json".format(os.path.basename(filepath).split(".")[0])

    # Writing JSON data to the file
    with open(filepath, "w+") as f:
        json.dump(output_json, f, indent=4)
                
    return segment_list, output_json
    
    
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

# def non_max_suppression(boxes, scores, max_output_size=1000, iou_threshold=0.1, score_threshold=float('-inf')):
#     # Combine boxes and scores into detections (each detection is a tuple of box and score)
#     detections = [(box, score) for box, score in zip(boxes, scores) if score > score_threshold]
    
#     # Sort detections by score in descending order
#     detections.sort(key=lambda x: x[1], reverse=True)
    
#     # Initialize selected boxes list
#     selected_boxes = []
    
#     # Loop over detections
#     for box, score in detections:
#         # Check if max_output_size is reached
#         if len(selected_boxes) >= max_output_size:
#             break
        
#         # Flag to determine if the box is selected
#         keep_box = True
        
#         # Compare box with previously selected boxes
#         for selected_box in selected_boxes:
#             # Calculate IoU (Intersection over Union) between the box and selected_box
#             intersect_xmin = max(box[0], selected_box[0])
#             intersect_ymin = max(box[1], selected_box[1])
#             intersect_xmax = min(box[2], selected_box[2])
#             intersect_ymax = min(box[3], selected_box[3])
#             intersect_width = max(0, intersect_xmax - intersect_xmin)
#             intersect_height = max(0, intersect_ymax - intersect_ymin)
#             intersect_area = intersect_width * intersect_height
            
#             box_area = (box[2] - box[0]) * (box[3] - box[1])
#             selected_box_area = (selected_box[2] - selected_box[0]) * (selected_box[3] - selected_box[1])
#             union_area = box_area + selected_box_area - intersect_area
            
#             iou = intersect_area / union_area
            
#             # If IoU is greater than threshold, suppress the box
#             if iou > iou_threshold:
#                 keep_box = False
#                 break
        
#         # If the box is not suppressed, add it to selected_boxes
#         if keep_box:
#             selected_boxes.append(box)
    
#     return selected_boxes


def scannedPageOCR(pdf_path):
    
    # Check if directories 'pages' and 'temp' exist, if not, create them
    if not os.path.exists('pages'):
        os.makedirs('pages')
    if not os.path.exists('temp'):
        os.makedirs('temp')

    images = pdf2img(pdf_path)
    text_content = ""
    
    # List of Dict
    output_json = []

    # looping over the images stored in pages directory
    for img in range(len(images)):
        curr_page_text_content = ""
        bbox = []
        individual_text = []
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
        text_blocks = lp.Layout([b for b in layout if b.type=='Text' or b.type=='Title' or b.type=='List'])
        h, w = image.shape[:2]

        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

        left_blocks = text_blocks.filter_by(left_interval, center=True)
        left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
        # The b.coordinates[1] corresponds to the y coordinate of the region
        # sort based on that can simulate the top-to-bottom reading order 
        right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
        right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

        # And finally combine the two lists and add the index
        text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

        ocr_agent = lp.TesseractAgent(languages='eng') 
        for block in text_blocks:
            print("BLOCK : ", block)
            x1 = int(block.block.x_1)
            y1 = int(block.block.y_1)
            x2 = int(block.block.x_2)
            y2 = int(block.block.y_2)
            bbox.append({"top": y1, "left": x1, "width": x2-x1, "height": y2-y1})
            segment_image = (block
                            .pad(left=5, right=5, top=5, bottom=5)
                            .crop_image(image))
                # add padding in each image segment can help
                # improve robustness 
                
            text = ocr_agent.detect(segment_image)
            individual_text.append(str(text))
            block.set(text=text, inplace=True)
        for txt in text_blocks.get_texts():
            text_content += txt[:-1]
            curr_page_text_content += txt[:-1]


        # Table Extraction
        extracted_tables = set()
        all_tables_df_4 =  pd.DataFrame()
        table_count = 1  # to maintain the count of table on individual page
        for l in layout:
            if l.type == 'Table' or l.type=='Figure':  # in some pdfs tables might be detectable as figure
                x_1 = int(l.block.x_1)
                y_1 = int(l.block.y_1)
                x_2 = int(l.block.x_2)
                y_2 = int(l.block.y_2)
                bbox.append({"top": y_1, "left": x_1, "width": x_2-x_1, "height": y_2-y_1})
                table_flag = 0
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
                                x_h, x_v = 0,int(box[0][0])
                                y_h, y_v = int(box[0][1]),0
                                width_h,width_v = image_width, int(box[2][0]-box[0][0])
                                height_h,height_v = int(box[2][1]-box[0][1]),image_height

                                horiz_boxes.append([x_h,y_h,x_h+width_h,y_h+height_h])
                                vert_boxes.append([x_v,y_v,x_v+width_v,y_v+height_v])

                            horiz_out = tf.image.non_max_suppression(
                                horiz_boxes,
                                probabilities,
                                max_output_size = 1000,
                                iou_threshold=0.1,
                                score_threshold=float('-inf'),
                                name=None
                            )
                            horiz_lines = np.sort(np.array(horiz_out))

                            vert_out = tf.image.non_max_suppression(
                                vert_boxes,
                                probabilities,
                                max_output_size = 1000,
                                iou_threshold=0.1,
                                score_threshold=float('-inf'),
                                name=None
                            )
                            vert_lines = np.sort(np.array(vert_out))

                            out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
                            unordered_boxes = []

                            for i in vert_lines:
                                unordered_boxes.append(vert_boxes[i][0])
                            ordered_boxes = np.argsort(unordered_boxes)

                            def intersection(box_1, box_2):
                                return [box_2[0], box_1[1],box_2[2], box_1[3]]

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

                            for i in range(len(horiz_lines)):
                                for j in range(len(vert_lines)):
                                    resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]] )
                                    for b in range(len(boxes)):
                                        the_box = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
                                        if(iou(resultant,the_box)>0.1):
                                            out_array[i][j] = texts[b]

                            out_array=np.array(out_array)
                            print("Table Content: " , out_array)
                            individual_text.append(str(out_array))
                            table_flag = 1
                            table = tuple(map(tuple, out_array))  # Convert to hashable tuple
                            if table not in extracted_tables:  # Check if table is unique
                                extracted_tables.add(table)
                                page_df = pd.DataFrame(out_array)
                                all_tables_df_4 = pd.concat([all_tables_df_4, page_df], ignore_index=True)
                                table_count+=1
                if table_flag == 0:
                    individual_text.append("")
        text_content += "\n"
        curr_page_text_content += "\n"
        output_string = StringIO()
        all_tables_df_4.to_csv(output_string, sep='\t', index=False)
        formatted_dataframe_string = output_string.getvalue()
        text_content += formatted_dataframe_string
        curr_page_text_content += formatted_dataframe_string
        meta_data = {'title': "", 'section_header': "", 'page': img+1, 'bbox': bbox, "individual_text": individual_text}
        output_json.append({"raw_text" : curr_page_text_content , "meta_data" : meta_data})

    # Store text_content
    os.makedirs("./test_output", exist_ok=True)
    with open(
        os.path.join("./test_output", os.path.basename(pdf_path).split(".")[0] + ".txt"),
        "w+",
    ) as f:
        f.write(text_content)

    # Creating the directory if it doesn't exist
    os.makedirs("./test_output", exist_ok=True)

    # Filepath of the JSON file you want to write
    pdf_path = "./test_output/{}.json".format(os.path.basename(pdf_path).split(".")[0])

    # Writing JSON data to the file
    with open(pdf_path, "w+") as f:
        json.dump(output_json, f, indent=4)

    return text_content, output_json



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_dir", help="Directory containing PDFs", required=True)
    args = parser.parse_args()

    docs = [args.doc_dir] if args.doc_dir.endswith(".pdf") else glob.glob(
        os.path.join(args.doc_dir, f"**/*.pdf"), recursive=True)

    for doc in docs:
        if(check_pdf(doc)):
            parser_extracted_text, parser_meta_data = pdf_parser(doc)
        else:
            extracted_text, meta_data = scannedPageOCR(doc)
    

