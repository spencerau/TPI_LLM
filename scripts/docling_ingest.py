"""
Convert documents from "Grounding Materials" into markdown files
and place them in the appropriate tpi_documents subdirectory.

Usage:
    python scripts/docling_ingest.py
    python scripts/docling_ingest.py --source "/path/to/Grounding Materials " --target /path/to/tpi_documents
"""

import argparse
import re
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter

FOLDER_MAP = {
    "1": "application_interview_process",
    "2": "legal_policy",
    "3": "best_practices_supporting_pwd_workplace",
    "4": "workplace_partnerships",
    "5": "evaluation_impact_continuousImprovement",
}

SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".html", ".htm", ".png", ".jpg", ".jpeg"}


def clean_stem(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()


def folder_number(folder_name: str) -> str | None:
    m = re.match(r"Folder\s+(\d+)", folder_name, re.IGNORECASE)
    return m.group(1) if m else None


def convert_file(converter: DocumentConverter, src: Path, dest_dir: Path) -> bool:
    out_name = clean_stem(src.stem) + ".md"
    out_path = dest_dir / out_name

    if out_path.exists():
        print(f"  skip (exists): {out_path.name}")
        return True

    try:
        result = converter.convert(str(src))
        md = result.document.export_to_markdown()
        if not md.strip():
            print(f"  warn (empty output): {src.name}")
            return False
        out_path.write_text(md, encoding="utf-8")
        print(f"  ok: {src.name} -> {out_path.name}")
        return True
    except Exception as e:
        print(f"  fail: {src.name} -- {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).parent.parent
    parser.add_argument(
        "--source",
        default="/Users/spencerau/Desktop/Grounding Materials ",
        help="Path to the Grounding Materials folder",
    )
    parser.add_argument(
        "--target",
        default=str(project_root / "tpi_documents"),
        help="Path to tpi_documents output directory",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)

    if not source.exists():
        sys.exit(f"Source not found: {source}")

    converter = DocumentConverter()

    total = ok = skip = fail = 0

    for folder in sorted(source.iterdir()):
        if not folder.is_dir():
            continue

        num = folder_number(folder.name)
        if num not in FOLDER_MAP:
            print(f"Unrecognized folder (skipping): {folder.name}")
            continue

        dest_dir = target / FOLDER_MAP[num]
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Folder {num}] {folder.name}")
        print(f"  -> {dest_dir}")

        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in SUPPORTED:
                continue
            total += 1
            result = convert_file(converter, f, dest_dir)
            if result:
                ok += 1
            else:
                fail += 1

    print(f"\nDone: {ok} converted, {fail} failed, {total} total")


if __name__ == "__main__":
    main()
