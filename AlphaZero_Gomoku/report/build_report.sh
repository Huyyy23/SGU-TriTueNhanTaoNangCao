#!/usr/bin/env bash
set -eu
# Run from the AlphaZero_Gomoku directory. The two output files are overwritten.
python_bin="./.venv/Scripts/python.exe"
soffice="C:/Program Files/LibreOffice/program/soffice.exe"
profile="${LOCALAPPDATA//\\//}/Temp/gomoku-report-export"
rm -f report/_indexes.json
trap 'rm -f report/_indexes.json' EXIT
for attempt in 1 2 3 4 5; do
    "$python_bin" report/build_report.py --build
    rm -f report/bao_cao.pdf
    "$soffice" "-env:UserInstallation=file:///$profile" --headless --convert-to pdf --outdir report report/bao_cao.docx
    test -f report/bao_cao.pdf
    if "$python_bin" report/build_report.py --index; then
        echo "DOCX và PDF đã có mục lục với số trang ổn định."
        exit 0
    else
        status=$?
        if [ "$status" -ne 3 ]; then exit "$status"; fi
    fi
done
echo "Không ổn định số trang sau năm lần kết xuất." >&2
exit 1
