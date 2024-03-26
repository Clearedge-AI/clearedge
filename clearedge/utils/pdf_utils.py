from PIL import Image
from doctr.io import DocumentFile
from pdf2image import convert_from_bytes
import pytesseract
import numpy as np

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
    bbox = bbox_data["metadata"]["bbox"]
    (first_group if bbox["left"] < first_third_width else second_group).append(bbox_data)

  sorted_output = sorted(first_group, key=lambda x: x["metadata"]["bbox"]["top"]) + \
      sorted(second_group, key=lambda x: x["metadata"]["bbox"]["top"])
  return sorted_output

def process_images(img_list, mydict, ocr_model, predictor=None, rapid_ocr=None, table_engine=None):
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
      metadata = {
        "type": label,
        "title": last_title,
        "section_header": last_subheading,
        "page": page_no,
        "bbox": {"top": ymin, "left": xmin, "width": xmax - xmin, "height": ymax - ymin}
      }
      temp_output.append({"raw_text": raw_text, "metadata": metadata})
    sorted_output = sort_and_update_metadata(temp_output, width)
    for bbox_data in sorted_output:
      metadata = bbox_data["metadata"]
      if metadata["type"] == "Title":
        last_title = bbox_data["raw_text"]
      elif metadata["type"] == "Section-header":
        last_subheading = bbox_data["raw_text"]
      bbox_data["metadata"]["title"] = last_title
      bbox_data["metadata"]["section_header"] = last_subheading
      output.append(bbox_data)
  return output
