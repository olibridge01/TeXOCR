
# LaTeX Equation Renderer

This script takes a `.txt` file of LaTeX equations (one equation per line) and renders each equation as a `.png` image using LaTeX and `dvipng`. It supports multiprocessing for faster rendering and tracks equations that fail to render.

## Features
- Converts LaTeX equations from `.txt` to `.png` images.
- Handles long equations with adjustable page width and scaling.
- Parallel processing with progress bar using `multiprocessing` and `tqdm`.
- Logs failed equations to a separate file for easy debugging.

## Requirements
This script requires the following:
1. **Python** (>= 3.7)
2. The following Python libraries:
   - `tqdm`
3. The following system dependencies:
   - `latex`
   - `dvipng`

## Installation Instructions

### 1. System Dependencies
Ensure that LaTeX and `dvipng` are installed on your system. Use the following commands depending on your operating system:

#### **Ubuntu/Debian**
```bash
sudo apt update
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng
```

#### **MacOS** (via Homebrew)
```bash
brew install mactex-no-gui
brew install dvipng
```

#### **Windows**
1. Install [MikTeX](https://miktex.org/download).
2. Ensure `latex` and `dvipng` are added to your PATH during installation.

### 2. Python Dependencies
#### **Option 1: Using `pip`**
Create and activate a new virtual environment, then install the required Python libraries:
```bash
python -m venv venv
source venv/bin/activate    # Use `venv\Scripts\activate` on Windows
pip install tqdm
```

#### **Option 2: Using `requirements.txt`**
If you prefer, you can create a `requirements.txt` file with the following content:
```
tqdm
```

Then install the dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your `.txt` file with LaTeX equations (one equation per line) in the working directory. For example:
   ```
   equations.txt
   ```

2. Run the script:
   ```bash
   python render_latex.py
   ```

3. Outputs:
   - Rendered images will be saved in the `rendered_equations` directory.
   - A `failed_equations.txt` file will log indices of equations that failed to render.

### Optional Arguments
You can modify the script to adjust parameters such as:
- **Output DPI**: Adjust the resolution of the output PNGs.
- **Number of Processes**: Specify the number of CPU cores to use for parallel rendering.

## Troubleshooting
- Ensure `latex` and `dvipng` are correctly installed and added to your PATH.
- Check the `failed_equations.txt` file for indices of problematic equations.
- Look at the corresponding `.log` files in the output directory for detailed LaTeX errors.

## License
This project is open-source and available under the [MIT License](LICENSE).
