class Metadata:
  def __init__(self, title=None, sub_heading=None, page_no=None, bbox=None, filename=None, doc_type=None, chunk_type=None):
    self.title = title
    self.sub_heading = sub_heading
    self.page_no = page_no
    self.bbox = bbox
    self.filename = filename
    self.doc_type = doc_type
    self.chunk_type = chunk_type

  def to_dict(self):
    """Returns a dictionary representation of the metadata fields."""
    return {
      "title": self.title,
      "sub_heading": self.sub_heading,
      "page_no": self.page_no,
      "bbox": self.bbox,
      "filename": self.filename,
      "doc_type": self.doc_type,
      "chunk_type": self.chunk_type
    }
