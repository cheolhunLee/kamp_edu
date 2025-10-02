"""
FastMCP Server Example (SSE 지원)
데스크톱 디렉토리 리소스와 스크린샷 도구를 제공하는 MCP 서버
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import io
import json

# MCP 서버 설정
mcp = FastMCP(
    "DesktopScreenshotServer",
    instructions="데스크톱 파일 목록 조회와 스크린샷 촬영 기능을 제공합니다.",
    host="0.0.0.0",
    port=8006,
)

# === RESOURCES ===
@mcp.resource("dir://current")
async def current_directory() -> str:
    """List the files in the current working directory"""
    current_path = Path.cwd()
    files = [str(f) for f in current_path.iterdir()]
    result = {
        "path": str(current_path),
        "files": files,
        "count": len(files)
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.resource("greeting://{name}")
async def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    result = {"message": f"Hello, {name}!", "name": name}
    return json.dumps(result, ensure_ascii=False, indent=2)

# === TOOLS ===
@mcp.tool()
async def add(a: int, b: int) -> str:
    """Add two numbers"""
    result = {"operation": f"{a} + {b}", "result": a + b}
    return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.tool()
async def take_screenshot() -> str:
    """Take a screenshot of the user's screen"""
    import pyautogui
    import base64
    
    buffer = io.BytesIO()
    screenshot = pyautogui.screenshot()
    screenshot.convert("RGB").save(buffer, format="JPEG", quality=60, optimize=True)
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    result = {
        "format": "jpeg",
        "size_bytes": len(buffer.getvalue()),
        "image_base64": image_data
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("🖥️  MCP Documents & Screenshot Server")
    print(f"📡 서버 주소: 0.0.0.0:8006")
    print("⏹️  서버 종료: Ctrl+C")
    
    mcp.run(transport="sse")