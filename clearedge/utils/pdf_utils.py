from collections import defaultdict

import re
import fitz

first_line_end_thesh = 0.8

def check_pdf(filepath):
  """
  Module to check if PDF is text extractable or not

  return bool
  True - if PDF text is extractable
  False - if PDF needs OCR parser
  """
  doc = fitz.open(filepath)
  num_pages = doc.page_count

  count = min(num_pages, 5)

  content_counter = 0

  for idx in range(count):
    page = doc.load_page(idx)

    # If page has content, increase content counter by 1
    word_data = page.get_text_words()

    if len(word_data):
      content_counter += 1

  print(f"Checking text extractable from PDF: {content_counter}")

  if content_counter:
    return True
  else:
    return False


block_miss = defaultdict(list)
block_double = defaultdict(list)


def compare_span(span1, span2):
  return all([span1[key] == span2[key] for key in ['font', 'flags', 'color']]) and abs(span1["size"] - span2["size"]) < 0.3


def font_hash(span):
  return f'{span["font"]}|{round(span["size"], 1)}|{span["flags"]}|{span["color"]}'


def merge(block1, block2):
  lines = block1["lines"].copy()
  lines.extend(block2["lines"])
  block2["lines"] = lines
  return block2


def coherent_check(line1, line2, strict=False):
  if line2['spans'][0]['text'][0].lower() != line2['spans'][0]['text'][0] and line1['spans'][-1]['text'][-1] in ['.', '?', '!']:
    return False
  if strict and (
    line2['spans'][0]['text'][0].lower() !=
    line2['spans'][0]['text'][0] or line1['spans'][-1]['text'][-1] in ['.', '?', '!']
  ):
    return False
  return True


def is_same_line(line1, line2):
  line1, line2 = (line2, line1) if line2['bbox'][1] < line1['bbox'][1] else (
    line1, line2
  )
  if line2['bbox'][1] < line1['bbox'][3]:
    return (line1['bbox'][3] - line2['bbox'][1]) / (line2['bbox'][3] - line1['bbox'][1]) > 0.8


def check_merge(block1, block2, line_gaps):
  # ensure the last line of first block doesn't end before the threshold
  total_width = block1["bbox"][2] - block1["bbox"][0]
  line1 = block1["lines"][-1]
  line2 = block2["lines"][0]
  if (line1["bbox"][2] - block1["bbox"][0]) / total_width < first_line_end_thesh:
    if not is_same_line(line1, line2):
      return False

  common_span = False
  common_spans = []
  # compare spans of same font in the text
  for span1 in line1["spans"]:
    span1_hash = font_hash(span1)
    for span2 in line2["spans"]:
      span2_hash = font_hash(span2)
      if compare_span(span1, span2) and not block2["list_item_start"]:
        gap = line2["bbox"][1] - line1["bbox"][3]
        gap = round(gap, 1)
        # joining blocks from different page
        if line1["word_bbox"][0][4] != line2["word_bbox"][0][4]:
          return coherent_check(line1, line2, strict=True)
        # joining black from different column
        elif line1["bbox"][3] > line2["bbox"][1]:
          return True if is_same_line(line1, line2) else coherent_check(line1, line2, strict=True)
        # # join block with fonts not merged before, ex in equidistance docs
        # elif span1_hash not in line_gaps:
        #     return True
        # join if block gap is same as line gap
        elif span1_hash in line_gaps and any([abs(line_gap - gap) < 0.2 for line_gap in line_gaps[span1_hash]]):
          return True
        common_span = True
        common_spans.append((f"{span1_hash}~{gap}"))

  common_spans = list(set(common_spans))

  if common_span and font_hash(line1["spans"][-1]) == font_hash(line2["spans"][0]):
    for span_hash_gap in common_spans:
      span_hash, gap = span_hash_gap.split('~')
      if float(span_hash.split('|')[1]) * 1.5 > float(gap):
        if len(block1['lines']) == 1 and len(block2['lines']) == 1:
          for gaps_offset in range(-3, 4):
            new_span_hash_gap = f"{span_hash}~{round(float(gap) + (gaps_offset / 10), 1)}"
            if block1['idx'] in block_miss[new_span_hash_gap]:
              block_double[new_span_hash_gap].append(block1['idx'])
            else:
              block_miss[new_span_hash_gap].append(block1['idx'])
            if block2['idx'] in block_miss[new_span_hash_gap]:
              block_double[new_span_hash_gap].append(block2['idx'])
            else:
              block_miss[new_span_hash_gap].append(block2['idx'])

  if not common_span:
    return is_same_line(line1, line2)

  return False


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


def detect_list_item_span(span):
  initial_token = span["text"].split(" ")[0]
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


def calculate_base_similarity(str1, str2):
  len1 = len(str1)
  len2 = len(str2)

  DP = [[0 for i in range(len1 + 1)]
        for j in range(2)]

  for i in range(0, len1 + 1):
    DP[0][i] = i

  for i in range(1, len2 + 1):
    for j in range(0, len1 + 1):
      if (j == 0):
        DP[i % 2][j] = i
      elif (str1[j - 1] == str2[i - 1]):
        DP[i % 2][j] = DP[(i - 1) % 2][j - 1]
      else:
        DP[i % 2][j] = (1 + min(DP[(i - 1) % 2][j],
                                min(DP[i % 2][j - 1],
                                DP[(i - 1) % 2][j - 1])))

  edit_distance = DP[len2 % 2][len1]
  base_similarity = (max(len1, len2) - edit_distance) / max(len1, len2)
  return (base_similarity)


def calculate_bbox_overlap(bb1, bb2):
  x_left = max(bb1[0], bb2[0])
  y_top = max(bb1[1], bb2[1])
  x_right = min(bb1[2], bb2[2])
  y_bottom = min(bb1[3], bb2[3])

  if x_right < x_left or y_bottom < y_top:
    return 0.0

  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  # compute the area of both AABBs
  bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
  bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

  overlap = intersection_area / float(bb1_area + bb2_area - intersection_area)
  return overlap


def calculate_similarity_scores(page_wise_candidates):
  for page_wise_lines in page_wise_candidates:
    for line in page_wise_lines:
      my_line = " ".join(line["tokens"])
      my_line = re.sub(r'[0-9]', '@', my_line)
      line["edited_text"] = my_line

  NCAND = 5
  rows, cols = (len(page_wise_candidates), NCAND)
  scores = [[0.0 for i in range(cols)] for j in range(rows)]
  WIN = 8
  weights = [1, 0.75, 0.5, 0.5, 0.5]
  for i in range(NCAND):
    for j in range(len(page_wise_candidates)):
      score = 0
      neighbours = range(max(j - WIN, 0), min(j + WIN, len(page_wise_candidates)))
      for k in neighbours:
        if k != j:
          if i < len(page_wise_candidates[k]) and i < len(page_wise_candidates[j]):
            base_similarity = calculate_base_similarity(page_wise_candidates[j][i]["edited_text"],
                                                        page_wise_candidates[k][i]["edited_text"])

            overlap = calculate_bbox_overlap(page_wise_candidates[j][i]["bbox"],
                                             page_wise_candidates[k][i]["bbox"])
            score += weights[i] * base_similarity * overlap
      scores[j][i] = score / len(neighbours)

  return scores


def remove_header_footer(page_wise_block_list, debug):
  page_wise_header_candidates = []
  page_wise_footer_candidates = []

  NCAND = 5
  MIN_SCORE = 0.4

  for blocks in page_wise_block_list:
    header_candidates = []
    footer_candidates = []

    for block in blocks:
      for line in block["lines"]:
        if len(header_candidates) >= NCAND:
          break
        header_candidates.append(line)

    for block in reversed(blocks):
      for line in reversed(block["lines"]):
        if len(footer_candidates) >= NCAND:
          break
        footer_candidates.append(line)

    page_wise_header_candidates.append(header_candidates)
    page_wise_footer_candidates.append(footer_candidates)

  # Calculate Similarity
  header_scores = calculate_similarity_scores(page_wise_header_candidates)
  footer_scores = calculate_similarity_scores(page_wise_footer_candidates)

  removed_headers = []
  removed_footers = []

  for j in range(len(page_wise_header_candidates)):
    i = 0
    for block in page_wise_block_list[j]:
      if i >= NCAND:
        break
      lines_to_remove = []
      for line in block["lines"]:
        if i >= NCAND:
          break
        if debug:
          print(header_scores[j][i], line)
        if header_scores[j][i] >= MIN_SCORE:
          removed_headers.append((j, line["edited_text"]))
          lines_to_remove.append(line)
        i = i + 1
      for line in lines_to_remove:
        block['lines'].remove(line)
    page_wise_block_list[j] = [block for block in page_wise_block_list[j] if len(block['lines'])]

  for j in range(len(page_wise_header_candidates)):
    i = 0
    for block in reversed(page_wise_block_list[j]):
      if i >= NCAND:
        break
      lines_to_remove = []
      for line in reversed(block["lines"]):
        if i >= NCAND:
          break
        if debug:
          print(footer_scores[j][i], line)
        if footer_scores[j][i] >= MIN_SCORE:
          removed_footers.append((j, line["edited_text"]))
          lines_to_remove.append(line)
        i = i + 1
      for line in lines_to_remove:
        block['lines'].remove(line)
    page_wise_block_list[j] = [block for block in page_wise_block_list[j] if len(block['lines'])]

  if debug:
    print("REMOVED HEADERS:")
    for header in removed_headers:
      print(header)

    print()

    print("REMOVED FOOTERS:")
    for footer in removed_footers:
      print(footer)

    print()

  return page_wise_block_list


def clean_blocks(blocks):
  """
  TODO: Remove Header and Footers from page_data
  TODO: detect tables and reconfigure block data accordingly
  """
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
