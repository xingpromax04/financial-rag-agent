"""用于财务指标精确计算的受限 Python 代码执行器。

运行环境：Python 3.11，Conda 环境 ``rag_311``。

安全边界：
    LLM 代码 -> AST 白名单校验 -> python -I 独立子进程
             -> 受限 builtins / import -> JSON 结果 -> 主进程

警告：该模块提供应用层多层防护，但不能替代操作系统级容器。
生产环境执行不可信用户代码时，仍应使用无网络、只读文件系统、
非 root 用户的 Docker / gVisor / Firecracker 容器。
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_ALLOWED_MODULES = {
    "decimal",
    "fractions",
    "math",
    "numpy",
    "pandas",
    "statistics",
}

_BANNED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

# 拦截文件、网络、进程、动态求值及原生内存相关入口。
_BANNED_ATTRIBUTES = {
    "ExcelFile",
    "HDFStore",
    "Popen",
    "ctypeslib",
    "dump",
    "dumps",
    "eval",
    "exec",
    "execv",
    "execve",
    "fork",
    "fromfile",
    "get_handle",
    "io",
    "load",
    "loads",
    "memmap",
    "open",
    "popen",
    "query",
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_gbq",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_stata",
    "read_table",
    "read_xml",
    "save",
    "savefig",
    "savetxt",
    "socket",
    "spawn",
    "system",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_gbq",
    "to_hdf",
    "to_html",
    "to_json",
    "to_latex",
    "to_markdown",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_stata",
    "to_xml",
}

_BANNED_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


class SandboxError(RuntimeError):
    """计算沙箱基础异常。"""


class CodeValidationError(SandboxError):
    """LLM 生成的代码不符合安全策略。"""


class CodeExecutionError(SandboxError):
    """子进程无法启动或返回无效结果。"""


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """计算沙箱资源与输出限制。"""

    timeout_seconds: float = 5.0  # 🔧【可调参数】超时后主进程强制终止子进程
    memory_limit_mb: int = 256  # 🔧【可调参数】Unix 使用 RLIMIT_AS，Windows 需容器补强
    max_code_chars: int = 20_000  # 🔧【可调参数】防止超大 AST 耗尽解析资源
    max_ast_nodes: int = 4_000  # 🔧【可调参数】限制代码结构复杂度
    max_input_bytes: int = 1_000_000  # 🔧【可调参数】JSON 输入上限
    max_output_bytes: int = 1_000_000  # 🔧【可调参数】JSON 结果与 stdout 上限

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        if self.memory_limit_mb < 64:
            raise ValueError("memory_limit_mb 不能小于 64")
        if min(
            self.max_code_chars,
            self.max_ast_nodes,
            self.max_input_bytes,
            self.max_output_bytes,
        ) < 1:
            raise ValueError("代码、AST、输入和输出上限必须大于 0")


@dataclass(slots=True)
class ExecutionResult:
    """沙箱执行结果。"""

    success: bool
    result: Any = None
    variables: dict[str, Any] = field(default_factory=dict)
    chart_data: Any = None
    stdout: str = ""
    error_type: str | None = None
    error_message: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转为可直接交给 Agent 或 API 层的字典。"""

        return asdict(self)


class _SafetyValidator(ast.NodeVisitor):
    """通过 AST 白名单和黑名单拦截危险语法。"""

    def __init__(self, max_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.node_count = 0
        self.errors: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.node_count += 1
        if self.node_count > self.max_nodes:
            self.errors.append(f"AST 节点数超过上限 {self.max_nodes}")
            return
        if isinstance(node, _BANNED_NODES):
            self.errors.append(
                f"第 {getattr(node, 'lineno', '?')} 行不允许 {type(node).__name__}"
            )
            return
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in _ALLOWED_MODULES:
                self.errors.append(
                    f"第 {node.lineno} 行不允许导入模块 {alias.name!r}"
                )
            if alias.asname and alias.asname.startswith("_"):
                self.errors.append(f"第 {node.lineno} 行导入别名不能以下划线开头")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or module not in _ALLOWED_MODULES:
            self.errors.append(f"第 {node.lineno} 行不允许从 {module!r} 导入")
        for alias in node.names:
            if (
                alias.name == "*"
                or alias.name.startswith("_")
                or alias.name in _BANNED_ATTRIBUTES
            ):
                self.errors.append(
                    f"第 {node.lineno} 行不允许导入 {alias.name!r}"
                )

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self.errors.append(f"第 {node.lineno} 行不允许访问 {node.id!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr in _BANNED_ATTRIBUTES:
            self.errors.append(f"第 {node.lineno} 行不允许访问属性 {node.attr!r}")
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALLS:
            self.errors.append(f"第 {node.lineno} 行不允许调用 {node.func.id!r}")
            return
        if isinstance(node.func, ast.Attribute):
            attribute = node.func.attr
            if attribute.startswith("read_") or attribute in _BANNED_ATTRIBUTES:
                self.errors.append(f"第 {node.lineno} 行不允许调用 {attribute!r}")
                return
        self.generic_visit(node)


_WORKER_SOURCE = r'''
import contextlib
import io
import json
import math
import sys
import types


def apply_resource_limits(memory_limit_mb, timeout_seconds):
    try:
        import resource
        memory_bytes = int(memory_limit_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        cpu_seconds = max(1, int(math.ceil(timeout_seconds)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    except (ImportError, AttributeError, OSError, ValueError):
        pass


def to_json_value(value, depth=0, max_items=1000):
    if depth > 8:
        return "<max-depth>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        return {
            str(key): to_json_value(item, depth + 1, max_items)
            for key, item in items
        }
    if isinstance(value, (list, tuple, set)):
        return [
            to_json_value(item, depth + 1, max_items)
            for item in list(value)[:max_items]
        ]
    if isinstance(value, types.ModuleType) or callable(value):
        return None

    module_name = type(value).__module__.split(".", maxsplit=1)[0]
    if module_name == "decimal" or module_name == "fractions":
        return str(value)
    if module_name == "numpy":
        if hasattr(value, "tolist"):
            return to_json_value(value.tolist(), depth + 1, max_items)
        if hasattr(value, "item"):
            return to_json_value(value.item(), depth + 1, max_items)
    if module_name == "pandas":
        if type(value).__name__ == "DataFrame":
            records = value.head(max_items).replace(
                {float("inf"): None, float("-inf"): None}
            )
            records = records.where(records.notna(), None).to_dict(orient="records")
            return {
                "type": "dataframe",
                "columns": [str(column) for column in value.columns],
                "records": to_json_value(records, depth + 1, max_items),
            }
        if type(value).__name__ in {"Series", "Index"}:
            return to_json_value(value.tolist(), depth + 1, max_items)
        if hasattr(value, "isoformat"):
            return value.isoformat()
    return str(value)


request = json.loads(sys.stdin.read())
apply_resource_limits(request["memory_limit_mb"], request["timeout_seconds"])

allowed_modules = set(request["allowed_modules"])
original_import = __import__


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_module = name.split(".", maxsplit=1)[0]
    if level or root_module not in allowed_modules:
        raise ImportError(f"module {name!r} is not allowed")
    return original_import(name, globals, locals, fromlist, level)


safe_builtins = {
    "__import__": safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

inputs = request["inputs"]
namespace = {"__builtins__": safe_builtins, **inputs}
input_names = set(inputs)
captured_stdout = io.StringIO()

try:
    with contextlib.redirect_stdout(captured_stdout):
        compiled = compile(request["code"], "<financial-calculation>", "exec")
        exec(compiled, namespace, namespace)

    variables = {}
    for name, value in namespace.items():
        if (
            name.startswith("_")
            or name in input_names
            or name in {"result", "chart_data"}
        ):
            continue
        if isinstance(value, types.ModuleType) or callable(value):
            continue
        variables[name] = to_json_value(value)

    payload = {
        "success": True,
        "result": to_json_value(namespace.get("result", variables)),
        "variables": variables,
        "chart_data": to_json_value(namespace.get("chart_data")),
        "stdout": captured_stdout.getvalue()[: request["max_output_chars"]],
        "error_type": None,
        "error_message": None,
    }
except BaseException as exc:
    payload = {
        "success": False,
        "result": None,
        "variables": {},
        "chart_data": None,
        "stdout": captured_stdout.getvalue()[: request["max_output_chars"]],
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:2000],
    }

sys.stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False))
'''


class CalculationEngine:
    """对 LLM 生成的财务计算代码进行校验和隔离执行。"""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def validate_code(self, code: str) -> None:
        """检查代码长度、语法树复杂度、导入和危险调用。"""

        if not isinstance(code, str) or not code.strip():
            raise CodeValidationError("计算代码不能为空")
        if len(code) > self.config.max_code_chars:
            raise CodeValidationError(
                f"代码长度超过上限 {self.config.max_code_chars}"
            )
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise CodeValidationError(
                f"代码语法错误：line={exc.lineno}, message={exc.msg}"
            ) from exc

        validator = _SafetyValidator(self.config.max_ast_nodes)
        validator.visit(tree)
        if validator.errors:
            unique_errors = list(dict.fromkeys(validator.errors))
            raise CodeValidationError("；".join(unique_errors[:10]))

    def execute(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """在独立子进程中执行计算并捕获 ``result`` / ``chart_data``。

        代码约定：
        - 将主计算结果赋给 ``result``；未赋值时返回所有新建普通变量。
        - 将画图所需的 x/y/series 数据赋给 ``chart_data``，由前端绘图。
        """

        self.validate_code(code)
        safe_inputs = self._validate_inputs(inputs or {})
        request = {
            "code": code,
            "inputs": safe_inputs,
            "allowed_modules": sorted(_ALLOWED_MODULES),
            "memory_limit_mb": self.config.memory_limit_mb,
            "timeout_seconds": self.config.timeout_seconds,
            "max_output_chars": self.config.max_output_bytes // 4,
        }
        serialized_request = json.dumps(
            request, ensure_ascii=False, allow_nan=False
        )
        if len(serialized_request.encode("utf-8")) > self.config.max_input_bytes:
            raise CodeValidationError(
                f"输入数据超过 {self.config.max_input_bytes} 字节"
            )

        started = time.perf_counter()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            with tempfile.TemporaryDirectory(prefix="rag_calc_") as temp_directory:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", _WORKER_SOURCE],
                    input=serialized_request,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=Path(temp_directory),
                    timeout=self.config.timeout_seconds,
                    check=False,
                    creationflags=creation_flags,
                )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - started) * 1000
            LOGGER.warning("calculation sandbox timed out: %.2f ms", duration_ms)
            return ExecutionResult(
                success=False,
                error_type="TimeoutError",
                error_message=(
                    f"计算超过 {self.config.timeout_seconds:.1f} 秒，子进程已终止"
                ),
                duration_ms=duration_ms,
            )
        except OSError as exc:
            raise CodeExecutionError("计算子进程启动失败") from exc

        duration_ms = (time.perf_counter() - started) * 1000
        stdout_bytes = len(completed.stdout.encode("utf-8"))
        if stdout_bytes > self.config.max_output_bytes:
            raise CodeExecutionError(
                f"沙箱输出超过 {self.config.max_output_bytes} 字节"
            )
        if completed.returncode != 0:
            message = completed.stderr.strip()[-2000:] or "子进程异常退出"
            raise CodeExecutionError(message)

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CodeExecutionError("沙箱返回了无效 JSON") from exc

        result = ExecutionResult(
            success=bool(payload.get("success")),
            result=payload.get("result"),
            variables=payload.get("variables") or {},
            chart_data=payload.get("chart_data"),
            stdout=str(payload.get("stdout") or ""),
            error_type=payload.get("error_type"),
            error_message=payload.get("error_message"),
            duration_ms=duration_ms,
        )
        LOGGER.info(
            "calculation sandbox finished: success=%s, duration_ms=%.2f",
            result.success,
            duration_ms,
        )
        return result

    @staticmethod
    def _validate_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        """确保变量名合法且输入可通过 JSON 跨进程传输。"""

        for name in inputs:
            if not isinstance(name, str) or not name.isidentifier():
                raise CodeValidationError(f"非法输入变量名：{name!r}")
            if name.startswith("_") or name in {"result", "chart_data", "__builtins__"}:
                raise CodeValidationError(f"输入变量名不可用：{name!r}")
        try:
            serialized = json.dumps(inputs, ensure_ascii=False, allow_nan=False)
            return json.loads(serialized)
        except (TypeError, ValueError) as exc:
            raise CodeValidationError("输入必须是可 JSON 序列化的有限数据") from exc


# 💡【面试加分点】“零幻觉计算”不是让 LLM 直接给出算术结果，
# 而是让 LLM 只负责生成可审计的公式代码，再由受限 Python 运行时执行。
# RAG 负责提供带来源的净利润、净资产等原始数据，沙箱负责确定性
# 计算、除零报错与结果序列化，最终同时保留数据证据和计算过程。
CalcEngine = CalculationEngine

__all__ = [
    "CalcEngine",
    "CalculationEngine",
    "CodeExecutionError",
    "CodeValidationError",
    "ExecutionResult",
    "SandboxConfig",
    "SandboxError",
]
