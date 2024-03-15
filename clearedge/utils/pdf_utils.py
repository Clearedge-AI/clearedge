from collections import defaultdict

import re
import fitz

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
