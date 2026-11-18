"""Error codes — design.md §1.2.9."""
from dataclasses import dataclass, field

from mcp.server.mcpserver.exceptions import ToolError  # anticipated tool failure: message reaches MCP clients

CODES: dict[int, tuple[str, bool]] = {  # code -> (name, retryable)
    -32700: ("ParseError", False), -32600: ("InvalidRequest", False),
    -32601: ("MethodNotFound", False), -32602: ("InvalidParams", False),
    -32603: ("InternalError", True), -32001: ("DeadlineExceeded", True),
    -32002: ("UpstreamUnavailable", True), -32003: ("RateLimited", True),
    -32011: ("TenantMissing", False), -32012: ("AclDenied", False),
    -32013: ("PiiBlocked", False), -32014: ("InjectionBlocked", False),
    -32015: ("HitlRequired", False), -32021: ("VersionConflict", True),
    -32031: ("LoopBudgetExceeded", False), -32032: ("CircuitOpen", True),
    -32041: ("UngroundedAnswer", False), -32042: ("ContextOverflow", False),
    -32051: ("ReplayedIdempotencyKey", False),
}

# HTTP status for the public API
HTTP_STATUS = {-32011: 401, -32012: 403, -32013: 422, -32014: 422, -32602: 422,
               -32003: 429, -32001: 504, -32002: 503, -32015: 202}


@dataclass
class AssistantError(ToolError):
    code: int
    message: str = ""
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:  # '[-32021] msg' survives MCP error text; McpClient parses it back
        return f"[{self.code}] {self.message}"

    @property
    def name(self) -> str:
        return CODES[self.code][0]

    @property
    def retryable(self) -> bool:
        return CODES[self.code][1]

    def to_dict(self) -> dict:
        d = {"code": self.code, "name": self.name, "message": self.message, "retryable": self.retryable}
        return {**d, "data": self.data} if self.data else d
