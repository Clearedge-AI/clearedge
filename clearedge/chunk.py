from .metadata import Metadata

class Chunk:
  """
  A class to represent a chunk of text and its associated metadata.
  """

  def __init__(self, text: str, metadata: Metadata):
    """
    Initializes a new instance of the Chunk class.

    Parameters:
    text (str): The text content of the chunk.
    metadata (dict): A dictionary containing metadata about the chunk.
    """
    self.text = text
    self.metadata = metadata

  def __repr__(self):
    """
    Returns a string representation of the Chunk object, including its text and metadata.

    Returns:
    str: A string representation of the Chunk object.
    """
    return f"Chunk(text={self.text}, metadata={self.metadata})"
