
# TeXOCR - Data Wrangling

This directory contains scripts for generating a rendered and split dataset from a `.txt` file containing LaTeX equations. This README provides installation instructions for the required non-Python dependencies.

### Prerequisites

The following non-Python tools are required:

- **`latex`**: For rendering LaTeX equations.
- **`dvipng`**: Converts DVI files (produced by LaTeX) to PNG images.
- **`imagemagick`**: For image processing (e.g., resizing).

### Installation

#### Linux (Ubuntu/Debian)
```bash
# Install LaTeX
sudo apt update
sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended

# Install dvipng
sudo apt install -y dvipng

# Install ImageMagick
sudo apt install -y imagemagick
```

#### macOS (Homebrew)
```bash
# Install LaTeX
brew install mactex-no-gui

# Install dvipng
brew install dvipng

# Install ImageMagick
brew install imagemagick
```

#### Windows
1. **LaTeX**: Install [MiKTeX](https://miktex.org/download), ensuring `dvipng` is included.
2. **ImageMagick**: Download from [ImageMagick official website](https://imagemagick.org/script/download.php).
