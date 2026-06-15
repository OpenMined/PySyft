"""Create test PDF dataset with manifest and readme."""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
PDF_DATASET_OUT_DIR = DATA_DIR / "PDFs"
SOURCE_PDF = DATA_DIR / "sample.pdf"


def create_pdfs():
    """Copy source PDF to 0000.pdf-0010.pdf (11 files)."""
    PDF_DATASET_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(11):
        dest = PDF_DATASET_OUT_DIR / f"{i:04d}.pdf"
        if not dest.exists():
            shutil.copy2(SOURCE_PDF, dest)
    return sorted(PDF_DATASET_OUT_DIR.glob("*.pdf"))


def create_manifest(pdf_files: list[Path]):
    """Create manifest.csv with filename, doc_id, and rotating category."""
    categories = ["report", "invoice", "contract", "memo"]
    csv_lines = ["filename,doc_id,category"]
    for i, pdf in enumerate(pdf_files):
        csv_lines.append(f"{pdf.name},DOC{i:05d},{categories[i % 4]}")
    manifest_path = DATA_DIR / "manifest.csv"
    manifest_path.write_text("\n".join(csv_lines))
    print(f"Created {manifest_path}")


def create_readme():
    """Create readme.md with usage instructions."""
    readme_path = DATA_DIR / "readme.md"
    readme_path.write_text(
        "10 PDFs created for testing. Use the dataset as:\n"
        "\n"
        'dataset_files = sc.resolve_dataset_files_path("pdfdata")\n'
        "print(len(dataset_files))\n"
    )
    print(f"Created {readme_path}")


if __name__ == "__main__":
    pdf_files = create_pdfs()
    print(f"Created {len(pdf_files)} PDFs in {PDF_DATASET_OUT_DIR}")
    create_manifest(pdf_files)
    create_readme()
