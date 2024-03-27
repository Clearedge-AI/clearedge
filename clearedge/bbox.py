class Bbox:
  def __init__(self, top, left, width, height):
    self.top = top
    self.left = left
    self.width = width
    self.height = height

  def to_dict(self):
    """Returns a dictionary representation of the bbox fields."""
    return {
      "top": self.top,
      "left": self.left,
      "width": self.width,
      "height": self.height
    }
