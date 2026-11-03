"""Scan publishable sources or a distribution archive for machine-specific information.

Only Python's standard library is used. PyTorch pickle metadata is inspected as
opcodes without unpickling objects, importing torch, or accessing a device.
"""
import argparse
import io
import ipaddress
import json
from pathlib import Path
import pickletools
import re
import tarfile
import zipfile

from check_release import source_files


PATTERNS = {
    "machine_path": re.compile(
        r"(?<![\w./:])/(?:home|Users|root|mnt|data|workspace|workplace|"
        r"vllm-workspace|run_deepseek|srv|opt|tmp|private|Volumes)/[^\s\"'`<>;,\])]+"
    ),
    "windows_user_path": re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]"),
    "internal_hostname": re.compile(r"\b[\w.-]+\.(?:internal|local|corp)\b", re.I),
    "email_address": re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "credential": re.compile(
        r"(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}|"
        r"\bAKIA[0-9A-Z]{16}\b|sk-[A-Za-z0-9_-]{20,}|"
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|"
        r"\b(?:https?|ssh)://[^\s/:@]+:[^\s/@]+@"
    ),
}
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")


class Scanner:
    def __init__(self):
        self.files = 0
        self.pickle_strings = 0
        self.findings = []

    def text(self, label, content):
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                self.findings.append({
                    "file": label, "kind": kind,
                    "line": content.count("\n", 0, match.start()) + 1,
                })
        for match in IPV4.finditer(content):
            try:
                ipaddress.IPv4Address(match.group())
            except ipaddress.AddressValueError:
                continue
            self.findings.append({"file": label, "kind": "ip_address"})

    def blob(self, label, data, depth=0):
        self.files += 1
        self.text(label + ":name", label)
        if depth > 8:
            self.findings.append({"file": label, "kind": "archive_nesting_limit"})
            return
        try:
            if zipfile.is_zipfile(io.BytesIO(data)):
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for entry in archive.infolist():
                        if entry.is_dir():
                            continue
                        nested = label + "!" + entry.filename
                        content = archive.read(entry)
                        if entry.filename.endswith(".pkl"):
                            self.text(nested + ":name", entry.filename)
                            for opcode, value, offset in pickletools.genops(content):
                                if isinstance(value, str):
                                    self.pickle_strings += 1
                                    self.text(f"{nested}:pickle@{offset}", value)
                        else:
                            self.blob(nested, content, depth + 1)
                return
            if label.endswith((".tar.gz", ".tgz", ".tar")):
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
                    for entry in archive:
                        if entry.issym() or entry.islnk():
                            self.findings.append({"file": label, "kind": "archive_link"})
                        elif entry.isfile():
                            with archive.extractfile(entry) as content:
                                self.blob(label + "!" + entry.name, content.read(), depth + 1)
                return
            self.text(label, data.decode("utf-8", errors="replace"))
        except (ValueError, OSError, EOFError, tarfile.TarError, zipfile.BadZipFile) as error:
            self.findings.append({"file": label, "kind": "scan_error", "error": type(error).__name__})

    def scan(self, path):
        if path.is_file():
            self.blob(path.name, path.read_bytes())
            return
        if not path.is_dir():
            raise ValueError("Provide an existing source directory or distribution file")
        for name in sorted({*source_files(path), "release/source_manifest.json"}):
            item = path / name
            if item.is_symlink():
                self.findings.append({"file": name, "kind": "source_symlink"})
            elif item.is_file():
                self.blob(name, item.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    scanner = Scanner()
    scanner.scan(args.path)
    print(json.dumps({
        "complete": not scanner.findings,
        "files_checked": scanner.files,
        "pickle_strings_checked": scanner.pickle_strings,
        "findings": scanner.findings,
    }, indent=2))
    raise SystemExit(bool(scanner.findings))


if __name__ == "__main__":
    main()
