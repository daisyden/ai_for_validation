#!/usr/bin/env python3

from __future__ import annotations

import shutil
from collections import Counter, OrderedDict
from pathlib import Path

from openpyxl import load_workbook


HOME = Path.home()
TARGET_XLSX = HOME / "opencode/classify/data/Non_inductor_ut_status_ww14_26.xlsx"
REFERENCE_XLSX = (
    HOME / "opencode/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx"
)
NOT_APPLIABLE_TXT = (
    HOME / "opencode/ai_for_validation/opencode/issue_triage/not_appliable.txt"
)
BACKUP_XLSX = TARGET_XLSX.with_name(TARGET_XLSX.stem + "_bk_before_classify_ut.xlsx")
TARGET_SHEET = "Non-Inductor XPU Skip"
REFERENCE_SHEETS = ("Not Appliable", "Not Applicable")
DEEP_ANALYSIS_SKILL = (
    HOME
    / "opencode/ai_for_validation/opencode/issue_triage/.claude/skills/bug_scrub/"
    "analyze_ci_result/check_xpu_case_existence/SKILL.md"
)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_column(ws, column_name: str) -> int:
    headers = [ws.cell(row=1, column=idx).value for idx in range(1, ws.max_column + 1)]
    for idx, value in enumerate(headers, start=1):
        if value == column_name:
            return idx
    for idx, value in enumerate(headers, start=1):
        if value is None:
            ws.cell(row=1, column=idx, value=column_name)
            return idx
    idx = ws.max_column + 1
    ws.cell(row=1, column=idx, value=column_name)
    return idx


def is_real_case_row(row_values: list[object]) -> bool:
    return any(value is not None for value in row_values[:23])


def infer_xpu_class(classname_cuda: object, classname_xpu: object) -> object:
    if classname_xpu not in (None, ""):
        return classname_xpu
    if classname_cuda in (None, ""):
        return classname_xpu
    value = str(classname_cuda)
    if value.endswith("CUDA"):
        return value[:-4] + "XPU"
    return value.replace("CUDA", "XPU")


def infer_xpu_name(name_cuda: object, name_xpu: object) -> object:
    if isinstance(name_xpu, str) and "xpugraphs" in name_xpu:
        name_xpu = None
    if name_xpu not in (None, ""):
        return name_xpu
    if name_cuda in (None, ""):
        return name_xpu
    value = str(name_cuda)
    if value.endswith("_cuda"):
        return value[:-5] + "_xpu"
    return value


def infer_xpu_file(testfile_cuda: object, testfile_xpu: object) -> object:
    if testfile_xpu not in (None, ""):
        return testfile_xpu
    return testfile_cuda


def set_if_changed(ws, row: int, column: int, value: object) -> bool:
    cell = ws.cell(row=row, column=column)
    if cell.value == value:
        return False
    cell.value = value
    return True


def get_reference_sheet(workbook):
    for sheet_name in REFERENCE_SHEETS:
        if sheet_name in workbook.sheetnames:
            return workbook[sheet_name]
    raise KeyError(f"Missing reference sheet from {REFERENCE_SHEETS}")


def collect_not_applicable_items() -> list[str]:
    target_wb = load_workbook(TARGET_XLSX, read_only=True)
    target_ws = target_wb[TARGET_SHEET]
    target_headers = [cell.value for cell in next(target_ws.iter_rows(min_row=1, max_row=1))]
    target_idx = {name: index for index, name in enumerate(target_headers) if name is not None}

    items: list[str] = []
    for row in target_ws.iter_rows(min_row=2, values_only=True):
        reason = normalize_text(row[target_idx["Reason"]])
        detail = normalize_text(row[target_idx["DetailReason"]])
        if reason.lower() == "not applicable" and detail:
            items.append(detail)

    reference_wb = load_workbook(REFERENCE_XLSX, read_only=True)
    reference_ws = get_reference_sheet(reference_wb)
    reference_headers = [
        cell.value for cell in next(reference_ws.iter_rows(min_row=1, max_row=1))
    ]
    reference_idx = {
        name: index for index, name in enumerate(reference_headers) if name is not None
    }
    operation_column = "Operation/API"
    if operation_column not in reference_idx:
        operation_column = "Torch Ops/API"

    for row in reference_ws.iter_rows(min_row=2, values_only=True):
        operation = normalize_text(row[reference_idx[operation_column]])
        if operation:
            items.append(operation)

    return list(OrderedDict.fromkeys(sorted(set(items), key=lambda item: item.lower())))


def write_not_appliable_file(items: list[str]) -> None:
    NOT_APPLIABLE_TXT.write_text("\n".join(items) + "\n")


def update_target_workbook() -> tuple[bool, int, Counter[str], list[tuple[int, str, str, str]]]:
    wb = load_workbook(TARGET_XLSX)
    ws = wb[TARGET_SHEET]

    explanation_col = ensure_column(ws, "Exaplaination")
    reason_tbd_col = ensure_column(ws, "Reason TBD")
    headers = [ws.cell(row=1, column=idx).value for idx in range(1, ws.max_column + 1)]
    idx = {name: index + 1 for index, name in enumerate(headers) if name is not None}

    dirty = False
    reason_counter: Counter[str] = Counter()
    pending_rows: list[tuple[int, str, str, str]] = []

    for row_idx in range(2, ws.max_row + 1):
        row_values = [
            ws.cell(row=row_idx, column=col_idx).value for col_idx in range(1, ws.max_column + 1)
        ]
        if not is_real_case_row(row_values):
            continue

        dirty |= set_if_changed(
            ws,
            row_idx,
            idx["testfile_xpu"],
            infer_xpu_file(
                ws.cell(row=row_idx, column=idx["testfile_cuda"]).value,
                ws.cell(row=row_idx, column=idx["testfile_xpu"]).value,
            ),
        )
        dirty |= set_if_changed(
            ws,
            row_idx,
            idx["classname_xpu"],
            infer_xpu_class(
                ws.cell(row=row_idx, column=idx["classname_cuda"]).value,
                ws.cell(row=row_idx, column=idx["classname_xpu"]).value,
            ),
        )
        dirty |= set_if_changed(
            ws,
            row_idx,
            idx["name_xpu"],
            infer_xpu_name(
                ws.cell(row=row_idx, column=idx["name_cuda"]).value,
                ws.cell(row=row_idx, column=idx["name_xpu"]).value,
            ),
        )

        reason = normalize_text(ws.cell(row=row_idx, column=idx["Reason"]).value)
        status_xpu = ws.cell(row=row_idx, column=idx["status_xpu"]).value
        reason_tbd = reason == ""
        dirty |= set_if_changed(ws, row_idx, reason_tbd_col, reason_tbd)
        reason_counter[reason or "<blank>"] += 1

        if status_xpu in (None, "") and reason_tbd:
            pending_rows.append(
                (
                    row_idx,
                    normalize_text(ws.cell(row=row_idx, column=idx["testfile_cuda"]).value),
                    normalize_text(ws.cell(row=row_idx, column=idx["classname_cuda"]).value),
                    normalize_text(ws.cell(row=row_idx, column=idx["name_cuda"]).value),
                )
            )
            if ws.cell(row=row_idx, column=explanation_col).value == "":
                dirty |= set_if_changed(ws, row_idx, explanation_col, None)

    if dirty:
        shutil.copy(TARGET_XLSX, BACKUP_XLSX)
        wb.save(TARGET_XLSX)

    return dirty, len(pending_rows), reason_counter, pending_rows


def main() -> None:
    if not TARGET_XLSX.exists():
        raise FileNotFoundError(TARGET_XLSX)
    if not REFERENCE_XLSX.exists():
        raise FileNotFoundError(REFERENCE_XLSX)

    items = collect_not_applicable_items()
    write_not_appliable_file(items)
    dirty, pending_count, reason_counter, pending_rows = update_target_workbook()

    print(f"target workbook: {TARGET_XLSX}")
    print(f"reference workbook: {REFERENCE_XLSX}")
    print(f"not appliable list: {NOT_APPLIABLE_TXT}")
    print(f"workbook_updated: {dirty}")
    if dirty:
        print(f"backup: {BACKUP_XLSX}")
    print(f"pending_deep_analysis_rows: {pending_count}")
    print("reason counts:")
    for reason, count in reason_counter.most_common():
        print(f"  {reason}: {count}")
    print("not appliable items:")
    for item in items:
        print(f"  - {item}")

    if pending_rows:
        print("pending rows requiring deep case existence analysis:")
        for row_idx, test_file, test_class, test_name in pending_rows:
            print(f"  - row {row_idx}: {test_file} | {test_class} | {test_name}")
        print("run the documented deep analysis workflow before filling Reason/DetailReason/Exaplaination:")
        print(f"  {DEEP_ANALYSIS_SKILL}")


if __name__ == "__main__":
    main()
