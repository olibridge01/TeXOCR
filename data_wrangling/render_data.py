import os
import logging
import argparse
import imagesize
import subprocess
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List
from multiprocessing import Pool, Manager, cpu_count

from TeXOCR.utils import load_config

def render_latex(args: Tuple[str, str, int, int, int, int, list]) -> None:
    """
    Renders a single LaTeX equation into a PNG file.

    Args:
        args (tuple): Contains (equation, output_dir, index, total_count, dpi).
    """
    equation, output_dir, index, total_count, dpi, patch_size, failed = args

    equation = equation.strip()
    if not equation:
        return

    # Determine the number of digits needed for zero padding
    zero_filled_index = str(index).zfill(len(str(total_count)))
    image_dir = Path(output_dir) / "images"
    os.makedirs(image_dir, exist_ok=True)

    base_filename = f"eq{zero_filled_index}"
    tex_file = Path(image_dir) / f"{base_filename}.tex"
    dvi_file = Path(image_dir) / f"{base_filename}.dvi"
    png_file = Path(image_dir) / f"{base_filename}.png"
    log_file = Path(image_dir) / f"{base_filename}.log"
    aux_file = Path(image_dir) / f"{base_filename}.aux"

    # Create LaTeX document
    tex_content = f"""
    \\documentclass[preview,border=1mm]{{standalone}}
    \\usepackage{{amsmath}}
    \\usepackage{{amsfonts}}
    \\usepackage{{amssymb}}
    \\usepackage[total={{16in, 8in}}]{{geometry}}
    \\begin{{document}}
    ${equation}$
    \\end{{document}}
    """

    # Write .tex file
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(tex_content)

    try:
        # Compile LaTeX to DVI
        subprocess.run(
            ["latex", "-interaction=nonstopmode", "-output-directory", image_dir, str(tex_file)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Convert DVI to PNG
        subprocess.run(
            ["dvipng", "-D", str(dpi), "-T", "tight", "-o", str(png_file), str(dvi_file)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to render equation {index} {equation}: {e}")
        failed.append((index, equation))
    finally:
        # Cleanup intermediate files
        tex_file.unlink(missing_ok=True)
        dvi_file.unlink(missing_ok=True)
        log_file.unlink(missing_ok=True)
        aux_file.unlink(missing_ok=True)

    # Resizing required for compatibility with the ViT encoder
    # Check if the file exists
    if png_file.exists():
        w, h = imagesize.get(str(png_file))
        new_h = h + (patch_size - h % patch_size) % patch_size

        w_interval = 2 * patch_size # Widths vary more, so increase the interval to reduce number of width 'bins'
        new_w = w + (w_interval - w % w_interval) % w_interval
        subprocess.run(
            ["convert", str(png_file), "-gravity", "center", "-extent", f"{new_w}x{new_h}", str(png_file)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

def render_images(input_file: str, output_dir: str, dpi: int = 300, num_processes: int = None, patch_size: int = 16) -> None:
    """
    Parallelizes LaTeX rendering using multiple processes.

    Args:
        input_file: Path to the .txt file containing LaTeX equations.
        output_dir: Directory to save the rendered images.
        dpi: Resolution of the output images.
        num_processes: Number of parallel processes to use. Defaults to the number of CPU cores.
        patch_size: Patch size for the image resizing.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read equations from the file
    with open(input_file, 'r', encoding='utf-8') as f:
        equations = [line.strip() for line in f if line.strip()]

    total_count = len(equations)
    num_processes = num_processes or cpu_count()

    # Prepare arguments for each process
    with Manager() as manager:
        failed = manager.list()
        tasks = [
            (equation, output_dir, i + 1, total_count, dpi, patch_size, failed)
            for i, equation in enumerate(equations)
        ]

        # Use a pool of workers
        with Pool(processes=num_processes) as pool:
            # Wrap the pool map with tqdm for a progress bar
            list(tqdm(pool.imap(render_latex, tasks), total=total_count, desc="Rendering"))

        print(f"Rendered {total_count} equations to {output_dir}, with {len(failed)} failures.")

        if failed:
            with open(Path(output_dir) / "failed.txt", "w", encoding="utf-8") as f:
                for index, equation in failed:
                    f.write(f"{index}: {equation}\n")

        # Create clean ids file
        with open(Path(output_dir) / "ids.txt", "w", encoding="utf-8") as f:
            for i, equation in enumerate(equations):
                zero_filled_index = str(i + 1).zfill(len(str(total_count)))
                f.write(f"eq{zero_filled_index}.png\n")

def prune_equations(input_file: str, failed_file: Path, output_dir: str):
    """
    Remove failed equations from the original equations file and save them to a separate file.
    
    Args:
        input_file: Path to the original .txt file containing LaTeX equations.
        failed_file: Path to the file containing failed equations.
        output_dir: Directory to save the pruned equations file.
    """
    if not failed_file.exists():
        return

    # Clean equations
    with open(failed_file, "r", encoding="utf-8") as f:
        failed_indices = {int(line.split(":")[0]) for line in f}

    with open(input_file, "r", encoding="utf-8") as f:
        equations = [line.strip() for i, line in enumerate(f, 1) if i not in failed_indices]

    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "labels_pruned.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(equations))

    # Clean ids
    with open(Path(output_dir) / "ids.txt", "r", encoding="utf-8") as f:
        ids = [line.strip() for i, line in enumerate(f, 1) if i not in failed_indices]

    with open(Path(output_dir) / "ids_pruned.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ids))

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Render LaTeX equations to images.")
    parser.add_argument("input_file", type=str, help="Path to the .txt file containing LaTeX equations.")
    parser.add_argument("output_dir", type=str, help="Directory to save the rendered images.")
    parser.add_argument("-c", "--config", type=str, default="config/data_config.yml", help="Path to the configuration file.")
    return parser.parse_args()

def main(args):
    """Main function to render LaTeX equations."""
    config = load_config(args.config)

    # Retrieve the necessary parameters from the config file
    input_file = args.input_file
    output_dir = args.output_dir
    dpi = config["dpi"]
    num_processes = config["num_processes"]
    patch_size = config["patch_size"]

    render_images(
        input_file=input_file,
        output_dir=output_dir,
        dpi=dpi,
        num_processes=num_processes,
        patch_size=patch_size
    )

    failed_file = Path(output_dir) / "failed.txt"
    prune_equations(
        input_file=input_file,
        failed_file=failed_file,
        output_dir=output_dir
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)