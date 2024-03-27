## Overview
Clearedge is a Python package designed to simplify the process of extracting raw text and metadata from documents. This powerful tool is capable of retrieving not only the text but also valuable metadata including titles, subheadings, page numbers, file names, bounding box (bbox) coordinates, and chunk types. Whether you're working on document analysis, data extraction projects, or building a RAG app with LLM, Clearedge provides a straightforward and efficient solution.


## Features

Text Extraction: Extract raw text from documents (currently supports pdf only. other file types coming soon).

Metadata Retrieval: Obtain metadata such as subheadings, page numbers, file names, bounding boxes and more.

Bounding Box Coordinates: Access bbox coordinates for text chunks, enabling spatial analysis of text placement within documents.

Chunk Type Identification: Identify types of text chunks (e.g., table, text and more) for advanced content analysis.

Support for Multiple Formats (coming soon): Compatible with popular document formats, ensuring broad usability.

## Installation

To install clearedge, you will need Python 3.6 or later. Installation is easy using pip:

```bash
pip install clearedge
```

## Quick Start
Here's a simple example to get you started with clearedge:
```python
from clearedge.reader.pdf import process_pdf

# Call the extractor with the path to your document
output = process_pdf('/path/to/your/document.pdf', use_ocr=True) # do not add use_ocr if you want to process documents faster. output is less accurate without ocr. 

# Extract text and metadata
text, metadata = output.text, output.metadata

# Accessing extracted text
print(text)

# Accessing metadata
print(metadata.to_dict())
```

## Documentation
For more detailed information on all the features and functionalities of Clearedge, please refer to the official documentation.

## Contributing
Contributions to Clearedge are welcome! If you have suggestions for improvements or bug fixes, please feel free to:
Open an issue to discuss what you would like to change.
Submit pull requests for us to review.

## License
Clearedge is released under the Apache 2.0 License. See the LICENSE file for more details.

## Acknowledgments
This project was inspired by the need for a simple, yet comprehensive tool for document analysis and metadata extraction. We thank all contributors and users for their support and feedback. Clearedge aims to be a valuable tool for developers, researchers, and anyone involved in processing and analyzing document content. We hope it simplifies your projects and helps you achieve your goals more efficiently.
