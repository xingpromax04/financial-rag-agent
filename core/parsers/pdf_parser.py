"""PDF 财务报告解析器，支持页眉页脚过滤与跨页表格拼接。

运行环境：Python 3.11，Conda 环境 ``rag_311``。
依赖：``pdfplumber>=0.11``、``pandas>=2.0``。
说明：pdfplumber 只解析 PDF 文本层，扫描版财报需先通过 OCR 生成文本层。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
import pdfplumber

PathLike: TypeAlias = str | Path
LOGGER = logging.getLogger(__name__)

_SPACE_RE = re.compile(r"[^\S\r\n]+")
_PAGE_NUMBER_RE = re.compile(r"^(?:第\s*)?\d+\s*(?:页)?(?:\s*/\s*\d+)?$")
_NUMBER_RE = re.compile(
    r"^[¥￥$]?[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?%?$|^\([\d,.]+\)$"
)
_DATE_RE = re.compile(r"^\d{4}[-/.\u5e74]\d{1,2}(?:[-/.\u6708]\d{1,2}日?)?$")

# 🔧【可调参数】表格识别敏感度。数值越大，越容易连接轻微错位的线段。
# ``vertical_strategy`` / ``horizontal_strategy`` 决定竖线和横线的来源：
# - lines：使用 PDF 中真实绘制的表格线，适合大多数带边框财务报表；
# - text：根据文字对齐推断边界，适合无框表，但需调整 min_words_* 参数。
# ``snap_tolerance`` 吸附临近线，``join_tolerance`` 拼接断线，
# ``intersection_tolerance`` 决定横竖线相交容差。extract_tables() 也使用同一组参数。
DEFAULT_TABLE_SETTINGS: dict[str, Any] = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
}


class PDFParseError(RuntimeError):
    """PDF 文本层或表格结构解析失败。"""


@dataclass(slots=True)
class PDFParseResult:
    """PDF 解析结果。``pages`` 与原 PDF 页码一一对应。"""

    pages: list[str]
    tables: list[pd.DataFrame]
    metadata: dict[str, Any]

    @property
    def text(self) -> str:
        """返回去除页眉页脚后的全文。"""

        return "\n\n".join(page for page in self.pages if page)


@dataclass(slots=True)
class _RawTable:
    """保留 DataFrame 构建前的表格行与页面几何信息。"""

    rows: list[list[str]]
    pages: list[int]
    table_index: int
    is_last_on_page: bool
    touches_top: bool
    touches_bottom: bool
    last_page: int = field(init=False)

    def __post_init__(self) -> None:
        self.last_page = self.pages[-1]


def _validate_pdf_path(file_path: PathLike) -> Path:
    path = Path(file_path).expanduser().resolve()
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"仅支持 .pdf 文件：{path}")
    if not path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{path}")
    return path


def clean_text(text: str | None, *, remove_page_number: bool = True) -> str:
    """清理页内空白，并过滤首尾位置的独立页码。"""

    if not text:
        return ""
    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if remove_page_number and lines and _PAGE_NUMBER_RE.fullmatch(lines[0]):
        lines.pop(0)
    if remove_page_number and lines and _PAGE_NUMBER_RE.fullmatch(lines[-1]):
        lines.pop()
    return "\n".join(lines)


def _clean_cell(value: Any) -> str:
    """将表格换行和不间断空格统一为普通空格。"""

    if value is None:
        return ""
    return " ".join(str(value).replace("\u00a0", " ").split())


def _clean_rows(raw_rows: list[list[Any]] | None) -> list[list[str]]:
    """删除空行、空列，并将不等长行补齐。"""

    rows = [[_clean_cell(cell) for cell in row] for row in (raw_rows or []) if row]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    keep_columns = [index for index in range(width) if any(row[index] for row in rows)]
    return [[row[index] for index in keep_columns] for row in rows]


def _unique_headers(values: list[str]) -> list[str]:
    """为空表头命名，并防止重名列破坏 DataFrame 查询。"""

    headers: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(values):
        base = value or f"column_{index + 1}"
        counts[base] = counts.get(base, 0) + 1
        suffix = f"_{counts[base]}" if counts[base] > 1 else ""
        headers.append(f"{base}{suffix}")
    return headers


def _normalized_row(row: list[str]) -> list[str]:
    return [re.sub(r"\W+", "", cell, flags=re.UNICODE).casefold() for cell in row]


def _same_row(left: list[str], right: list[str]) -> bool:
    return len(left) == len(right) and _normalized_row(left) == _normalized_row(right)


def _cell_kind(cell: str) -> str:
    """将单元格归类为空、数字、日期或文本，用于比较列布局。"""

    if not cell:
        return "empty"
    if _NUMBER_RE.fullmatch(cell.replace(" ", "")):
        return "number"
    if _DATE_RE.fullmatch(cell.replace(" ", "")):
        return "date"
    return "text"


def _layout_similarity(left: list[str], right: list[str]) -> float:
    """计算两行的列类型相似度，作为无重复表头时的续表证据。"""

    if len(left) != len(right) or not left:
        return 0.0
    matches = sum(
        left_kind == right_kind
        for left_kind, right_kind in zip(
            map(_cell_kind, left), map(_cell_kind, right), strict=True
        )
    )
    return matches / len(left)


def _can_merge(
    previous: _RawTable,
    current: _RawTable,
    *,
    min_layout_similarity: float,
) -> bool:
    """综合页码、页内位置、列数和列类型判断是否为跨页续表。"""

    is_adjacent_boundary = (
        previous.last_page + 1 == current.pages[0]
        and previous.is_last_on_page
        and current.table_index == 0
    )
    same_width = len(previous.rows[0]) == len(current.rows[0])
    if not is_adjacent_boundary or not same_width:
        return False

    if _same_row(previous.rows[0], current.rows[0]):
        return True
    if not (previous.touches_bottom and current.touches_top):
        return False
    similarity = _layout_similarity(previous.rows[-1], current.rows[0])
    return similarity >= min_layout_similarity


def _merge_across_pages(
    tables: list[_RawTable],
    *,
    min_layout_similarity: float,
) -> list[_RawTable]:
    """合并跨页续表，并删除后续页重复出现的表头。"""

    merged: list[_RawTable] = []
    for current in tables:
        if not merged or not _can_merge(
            merged[-1], current, min_layout_similarity=min_layout_similarity
        ):
            merged.append(current)
            continue

        previous = merged[-1]
        continuation = (
            current.rows[1:]
            if _same_row(previous.rows[0], current.rows[0])
            else current.rows
        )
        previous.rows.extend(continuation)
        previous.pages.extend(current.pages)
        previous.last_page = current.last_page
        previous.is_last_on_page = current.is_last_on_page
        previous.touches_bottom = current.touches_bottom
    return merged


def _to_dataframe(table: _RawTable, first_row_header: bool) -> pd.DataFrame:
    """将已合并的原始表格转为 DataFrame，并保留来源页码。"""

    if first_row_header:
        columns = _unique_headers(table.rows[0])
        data = [row for row in table.rows[1:] if not _same_row(table.rows[0], row)]
    else:
        columns = [f"column_{index + 1}" for index in range(len(table.rows[0]))]
        data = table.rows

    frame = pd.DataFrame(data, columns=columns)
    frame.replace({"": pd.NA}, inplace=True)
    frame.attrs["source_pages"] = table.pages.copy()
    return frame


def _validate_ratios(
    header_ratio: float,
    footer_ratio: float,
    edge_margin_ratio: float,
    min_layout_similarity: float,
) -> None:
    if not 0 <= header_ratio < 0.5 or not 0 <= footer_ratio < 0.5:
        raise ValueError("页眉和页脚裁剪比例必须在 [0, 0.5) 内")
    if header_ratio + footer_ratio >= 0.8:
        raise ValueError("页眉与页脚裁剪比例之和必须小于 0.8")
    if not 0 <= edge_margin_ratio <= 0.5:
        raise ValueError("跨页表格边界比例必须在 [0, 0.5] 内")
    if not 0 <= min_layout_similarity <= 1:
        raise ValueError("表格布局相似度阈值必须在 [0, 1] 内")


def parse_pdf(
    file_path: PathLike,
    *,
    password: str | None = None,
    header_ratio: float = 0.07,  # 🔧【可调参数】顶部 7% 区域视为页眉
    footer_ratio: float = 0.07,  # 🔧【可调参数】底部 7% 区域视为页脚
    edge_margin_ratio: float = 0.12,  # 🔧【可调参数】判定表格接近页边界的比例
    min_layout_similarity: float = 0.75,  # 🔧【可调参数】续表列类型相似度阈值
    table_settings: dict[str, Any] | None = None,  # 🔧【可调参数】表格线识别规则
    first_row_header: bool = True,  # 🔧【可调参数】是否将表格首行作为列名
    text_x_tolerance: float = 2,  # 🔧【可调参数】横向字符合并容差
    text_y_tolerance: float = 3,  # 🔧【可调参数】纵向文本行合并容差
) -> PDFParseResult:
    """提取 PDF 正文和表格，并智能合并相邻页上的续表。

    PDF 原生坐标系以左下角为原点，但 pdfplumber 同时提供 ``top`` /
    ``bottom`` 距页面顶部的坐标。``crop((x0, top, x1, bottom))`` 使用的
    就是这套自顶向下坐标，因此可以按页高比例稳定剔除页眉页脚。

    ``find_tables()`` 与 ``extract_tables()`` 使用相同的 ``table_settings``。
    前者还返回表格 bbox，这是判断表格是否触及页顶/页底的关键，
    所以本模块选择 find_tables() 后再调用 Table.extract()。
    """

    _validate_ratios(
        header_ratio, footer_ratio, edge_margin_ratio, min_layout_similarity
    )
    path = _validate_pdf_path(file_path)
    settings = {**DEFAULT_TABLE_SETTINGS, **(table_settings or {})}
    page_texts: list[str] = []
    raw_tables: list[_RawTable] = []
    metadata: dict[str, Any] = {}

    try:
        with pdfplumber.open(path, password=password) as pdf:
            metadata = dict(pdf.metadata or {})
            metadata["page_count"] = len(pdf.pages)

            for page_number, page in enumerate(pdf.pages, start=1):
                page_top = float(page.bbox[1])
                page_bottom = float(page.bbox[3])
                body_top = page_top + page.height * header_ratio
                body_bottom = page_bottom - page.height * footer_ratio
                body_height = body_bottom - body_top

                # 【功能原理】只在正文 bbox 内提取，页眉页脚不会进入 RAG 切块。
                body_bbox = (page.bbox[0], body_top, page.bbox[2], body_bottom)
                body_page = page.crop(body_bbox)
                page_text = body_page.extract_text(
                    x_tolerance=text_x_tolerance,
                    y_tolerance=text_y_tolerance,
                )
                page_texts.append(clean_text(page_text))

                found_tables = body_page.find_tables(table_settings=settings)
                valid_tables = [
                    (rows, found)
                    for found in found_tables
                    if (rows := _clean_rows(found.extract()))
                ]
                for table_index, (rows, found) in enumerate(valid_tables):
                    top = float(found.bbox[1])
                    bottom = float(found.bbox[3])
                    raw_tables.append(
                        _RawTable(
                            rows=rows,
                            pages=[page_number],
                            table_index=table_index,
                            is_last_on_page=table_index == len(valid_tables) - 1,
                            touches_top=(top - body_top)
                            <= body_height * edge_margin_ratio,
                            touches_bottom=(body_bottom - bottom)
                            <= body_height * edge_margin_ratio,
                        )
                    )
    except Exception as exc:
        raise PDFParseError(f"PDF 解析失败：{path}") from exc

    merged_tables = _merge_across_pages(
        raw_tables, min_layout_similarity=min_layout_similarity
    )
    tables = [_to_dataframe(table, first_row_header) for table in merged_tables]
    LOGGER.info(
        "PDF 解析完成：file=%s, pages=%s, raw_tables=%s, merged_tables=%s",
        path,
        len(page_texts),
        len(raw_tables),
        len(tables),
    )
    return PDFParseResult(pages=page_texts, tables=tables, metadata=metadata)


__all__ = [
    "DEFAULT_TABLE_SETTINGS",
    "PDFParseError",
    "PDFParseResult",
    "clean_text",
    "parse_pdf",
]
