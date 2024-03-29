## Overview
Clearedge is a Python package designed to simplify the process of extracting raw text and metadata from documents. You can use it to retrieve not only the text but also valuable metadata including titles, subheadings, page numbers, file names, bounding box (bbox) coordinates, and chunk types. Whether you're working on document analysis, data extraction projects, or building a RAG app with LLM, Clearedge provides a straightforward and efficient solution.


## Features

- Text Extraction: Extract raw text from documents (currently supports pdf only. other file types coming soon).
- Metadata Retrieval: Obtain metadata such as subheadings, page numbers, file names, bounding boxes and more.
- Bounding Box Coordinates: Access bbox coordinates for text chunks, enabling spatial analysis of text placement within documents.
- Chunk Type Identification: Identify types of text chunks (e.g., table, text and more) for advanced content analysis.
- Support for Multiple Formats (coming soon): Compatible with popular document formats, ensuring broad usability.

## Installation
### Prerequisites 

Python 3.9 (or higher) 
To install clearedge, you will need Python 3.9 or later. Installation is easy using pip:

Since we use Tesseract and Doctr, you will need extra dependencies. 

For MacOS users, you need to run: 
```shell
brew install tesseract cairo pango gdk-pixbuf libffi
```

For Windows users, those dependencies are included in GTK. You can find the latest installer over [here](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).

You can then install the latest release of the package using [pypi](https://pypi.org/project/clearedge/) as follows:
```bash
pip install clearedge
```

## Quick Start
Here's a simple example to get you started with clearedge:
```python
from clearedge.reader.pdf import process_pdf

# Call the extractor with the path to your document
chunks = process_pdf('/path/to/your/document.pdf', use_ocr=True) # do not add use_ocr for faster processing. output is less accurate without ocr. 

# Extract text and metadata
for chunk in chunks:
    text, metadata = chunk.text, chunk.metadata
    print(text) # Accessing extracted text
    print(metadata.to_dict()) # Accessing metadata
```

## Documentation
For more detailed information on all the features and functionalities of Clearedge, please refer to the official documentation (coming soon).

## Contributing
Contributions to Clearedge are welcome! If you have suggestions for improvements or bug fixes, please feel free to:
Open an issue to discuss what you would like to change.
Submit pull requests for us to review.

## Citation

If you wish to cite this project, feel free to use this [BibTeX](http://www.bibtex.org/) reference:

```bibtex
@misc{clearedge2024,
    title={clearedge: RAG preprocessor},
    author={Clearedge AI},
    year={2024},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/Clearedge-AI/clearedge}}
}
```

## License
Clearedge is released under the Apache 2.0 License. See the [`LICENSE`](https://github.com/Clearedge-AI/clearedge?tab=Apache-2.0-1-ov-file#readme) file for more details.

## Acknowledgments
This project was inspired by the need for a simple, yet comprehensive tool for document analysis and metadata extraction. We thank all contributors and users for their support and feedback. Clearedge aims to be a valuable tool for developers, researchers, and anyone involved in processing and analyzing document content. We hope it simplifies your projects and helps you achieve your goals more efficiently.
