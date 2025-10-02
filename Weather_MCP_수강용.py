"""
MCP Weather Server (SSE 지원)
실시간 날씨 정보를 제공하는 MCP 서버
"""

from mcp.server.fastmcp import FastMCP
import requests
import json

# =====  API 키 설정 (필수!) =====
WEATHER_API_KEY = ""  # WeatherAPI.com에서 발급받은 키를 여기에 입력하세요
# ================================

# MCP 서버 설정
mcp = FastMCP(
    "WeatherServer",
    instructions="실시간 날씨 정보를 조회합니다. 도시 이름을 입력하면 현재 날씨 데이터를 제공합니다.",
    host="0.0.0.0",
    port=8005,
)

# WeatherAPI URL 템플릿
WEATHER_URL = "https://api.weatherapi.com/v1/current.json?key=WEATHER_KEY&q=CITY_NAME"

@mcp.tool()
async def get_todays_weather(city_name: str) -> str:
    # API 요청 URL 생성
    url = WEATHER_URL.replace("WEATHER_KEY", WEATHER_API_KEY).replace(
        "CITY_NAME", city_name
    )
    # WeatherAPI 호출
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        # 성공적인 응답 처리
        data = response.json()
        # 필요한 날씨 정보만 추출
        weather_info = {
            "status": "success",
            "city": data.get("location", {}).get("name", city_name),
            "country": data.get("location", {}).get("country", "Unknown"),
            "region": data.get("location", {}).get("region", ""),
            "local_time": data.get("location", {}).get("localtime", ""),
            "temperature": {
                "celsius": data.get("current", {}).get("temp_c"),
                "fahrenheit": data.get("current", {}).get("temp_f"),
                "feels_like_c": data.get("current", {}).get("feelslike_c"),
                "feels_like_f": data.get("current", {}).get("feelslike_f")
            },
            "condition": {
                "text": data.get("current", {}).get("condition", {}).get("text"),
                "icon": data.get("current", {}).get("condition", {}).get("icon")
            },
            "wind": {
                "speed_kph": data.get("current", {}).get("wind_kph"),
                "speed_mph": data.get("current", {}).get("wind_mph"),
                "direction": data.get("current", {}).get("wind_dir"),
                "degree": data.get("current", {}).get("wind_degree")
            },
            "atmosphere": {
                "humidity": data.get("current", {}).get("humidity"),
                "pressure_mb": data.get("current", {}).get("pressure_mb"),
                "pressure_in": data.get("current", {}).get("pressure_in"),
                "visibility_km": data.get("current", {}).get("vis_km"),
                "visibility_miles": data.get("current", {}).get("vis_miles"),
                "uv_index": data.get("current", {}).get("uv")
            },
            "last_updated": data.get("current", {}).get("last_updated")
        }
        print(f"성공: {weather_info['city']}, {weather_info['temperature']['celsius']}°C")
        return json.dumps(weather_info, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    print("  MCP Weather Server (SSE)")
    print("=" * 50)
    print(f"📡 서버 주소: 0.0.0.0:8005")
    print(f"🔧 서버명: WeatherServer")
    
    print("\n 서버 종료: Ctrl+C")
    print("=" * 50)
    
    try:
        #  중요: SSE transport로 실행
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        print("\n 서버가 종료되었습니다.")
    except Exception as e:
        print(f"\n 서버 오류: {e}")