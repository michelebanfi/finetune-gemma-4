"""
LaTeX/TikZ compilation utility.

Compiles TikZ code (full LaTeX document) to PDF and converts to PNG.

System requirements:
  - pdflatex  (texlive-base or texlive-full)
  - pdftoppm  (poppler-utils) — used by pdf2image

Install on Linux/Mac:
  sudo apt-get install texlive-full poppler-utils   # Ubuntu/Debian
  brew install mactex poppler                        # macOS
"""
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CompileResult:
    success: bool
    png_path: Optional[str] = None   # Path to rendered PNG (first page)
    error_msg: Optional[str] = None  # Extracted error summary
    log: str = ""                    # Full pdflatex log


def _extract_error(log: str) -> str:
    """Pull the first LaTeX error line from the pdflatex log."""
    lines = log.splitlines()
    errors = [l for l in lines if l.startswith("!")]
    if errors:
        return "\n".join(errors[:5])
    # Fall back to last non-empty lines
    tail = [l for l in lines if l.strip()][-10:]
    return "\n".join(tail)


def compile_tikz(code: str, timeout: int = 30, dpi: int = 150) -> CompileResult:
    """
    Compile a full LaTeX/TikZ document string to a PNG image.

    Args:
        code:    Complete LaTeX document (\\documentclass ... \\end{document})
        timeout: Max seconds to allow pdflatex to run (TikZ can loop forever)
        dpi:     Resolution of the output PNG

    Returns:
        CompileResult with success status, PNG path (if succeeded), and log.
    """
    # Verify pdflatex is available
    if not shutil.which("pdflatex"):
        return CompileResult(
            success=False,
            error_msg="pdflatex not found. Install texlive: brew install mactex (Mac) or sudo apt-get install texlive-full (Linux)",
        )

    tmpdir = tempfile.mkdtemp(prefix="tikz_compile_")
    tex_path = os.path.join(tmpdir, "figure.tex")
    pdf_path = os.path.join(tmpdir, "figure.pdf")
    png_path = os.path.join(tmpdir, "figure.png")

    try:
        # Write the LaTeX source
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Run pdflatex (twice for proper cross-references, but once is usually enough for TikZ)
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory", tmpdir,
                tex_path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmpdir,
        )

        log = result.stdout + result.stderr

        if result.returncode != 0 or not os.path.exists(pdf_path):
            return CompileResult(
                success=False,
                error_msg=_extract_error(log),
                log=log,
            )

        # Convert PDF → PNG using pdf2image (requires poppler)
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf_path, dpi=dpi)
            if pages:
                pages[0].save(png_path, "PNG")
                # Copy to a persistent location outside tmpdir
                import uuid
                out_dir = os.path.join(tempfile.gettempdir(), "tikz_outputs")
                os.makedirs(out_dir, exist_ok=True)
                persistent_png = os.path.join(out_dir, f"{uuid.uuid4().hex}.png")
                shutil.copy(png_path, persistent_png)
                return CompileResult(success=True, png_path=persistent_png, log=log)
        except ImportError:
            # pdf2image not installed — still a success, just no PNG
            return CompileResult(success=True, log=log,
                                 error_msg="pdf2image not installed; PNG conversion skipped")
        except Exception as e:
            return CompileResult(success=True, log=log,
                                 error_msg=f"PNG conversion failed: {e}")

        return CompileResult(success=True, log=log)

    except subprocess.TimeoutExpired:
        return CompileResult(
            success=False,
            error_msg=f"pdflatex timed out after {timeout}s (possible infinite TikZ loop)",
            log="",
        )
    except Exception as e:
        return CompileResult(success=False, error_msg=str(e), log="")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def check_dependencies() -> dict:
    """Check which compilation dependencies are available."""
    status = {}

    status["pdflatex"] = shutil.which("pdflatex") is not None

    try:
        import pdf2image
        status["pdf2image"] = True
    except ImportError:
        status["pdf2image"] = False

    try:
        import PIL
        status["pillow"] = True
    except ImportError:
        status["pillow"] = False

    return status


if __name__ == "__main__":
    # Quick self-test
    test_code = r"""
\documentclass{standalone}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}
  \draw[thick,->] (0,0) -- (2,0) node[right] {$x$};
  \draw[thick,->] (0,0) -- (0,2) node[above] {$y$};
  \draw[blue,thick] (0,0) circle (1);
  \node at (0.7,0.7) {$r=1$};
\end{tikzpicture}
\end{document}
"""
    deps = check_dependencies()
    print("Dependencies:", deps)

    result = compile_tikz(test_code)
    print(f"Compilation success: {result.success}")
    if result.success:
        print(f"PNG saved to: {result.png_path}")
    else:
        print(f"Error: {result.error_msg}")
