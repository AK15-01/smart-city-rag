import requests, json, os, time
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

CITIES = {
    "杭州": {"lat": 30.2741, "lon": 120.1551, "bbox": "29.9,119.7,30.6,120.7"},
}
TARGET_CITY = "杭州"

def fetch_transit_stations(city_name):
    city = CITIES[city_name]
    bbox = city["bbox"]
    query = f"""
    [out:json][timeout:30];
    (
      node["railway"="station"]({bbox});
      node["railway"="subway_entrance"]({bbox});
      node["amenity"="bus_station"]({bbox});
    );
    out body;
    """
    print(f"[交通] 正在爬取 {city_name} 站点数据...")
    try:
        resp = requests.post("https://overpass-api.de/api/interpreter",
                             data={"data": query}, timeout=40)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        stations = []
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name") or tags.get("name:zh") or tags.get("name:en")
            if not name:
                continue
            stations.append({
                "id": el["id"], "name": name,
                "type": tags.get("railway") or tags.get("amenity", "站点"),
                "lat": el.get("lat"), "lon": el.get("lon"),
                "lines": tags.get("line") or tags.get("network", ""),
                "operator": tags.get("operator", ""), "city": city_name,
            })
        print(f"[交通] 获取到 {len(stations)} 个站点")
        return stations
    except Exception as e:
        print(f"[交通] 爬取失败: {e}")
        return []

def fetch_weather(city_name):
    city = CITIES[city_name]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city["lat"], "longitude": city["lon"],
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
        "timezone": "Asia/Shanghai", "forecast_days": 7,
    }
    print(f"[天气] 正在获取 {city_name} 天气数据...")
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        wmo = {0:"晴天",1:"基本晴朗",2:"局部多云",3:"阴天",45:"雾",
               61:"小雨",63:"中雨",65:"大雨",80:"阵雨",95:"雷雨"}
        current = raw.get("current", {})
        daily = raw.get("daily", {})
        weather = {
            "city": city_name,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "current": {
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "precipitation": current.get("precipitation"),
                "condition": wmo.get(current.get("weather_code", 0), "未知"),
            },
            "forecast": [],
        }
        for i, date in enumerate(daily.get("time", [])):
            weather["forecast"].append({
                "date": date,
                "max_temp": daily["temperature_2m_max"][i],
                "min_temp": daily["temperature_2m_min"][i],
                "precipitation": daily["precipitation_sum"][i],
                "condition": wmo.get(daily["weather_code"][i], "未知"),
            })
        print(f"[天气] 当前：{weather['current']['condition']}，{weather['current']['temperature']}°C")
        return weather
    except Exception as e:
        print(f"[天气] 获取失败: {e}")
        return {}

def save_and_chunk(stations, weather, city_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = os.path.join(DATA_DIR, f"{city_name}_data_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"stations": stations, "weather": weather}, f, ensure_ascii=False, indent=2)
    print(f"[保存] 原始数据 → {json_path}")

    chunks = []
    if weather:
        c = weather["current"]
        chunk = (f"【{city_name}天气实况】更新时间：{weather['updated_at']}\n"
                 f"当前天气：{c['condition']}，气温{c['temperature']}°C，"
                 f"湿度{c['humidity']}%，风速{c['wind_speed']}km/h\n")
        for fd in weather.get("forecast", []):
            chunk += f"{fd['date']}：{fd['condition']}，{fd['min_temp']}~{fd['max_temp']}°C，降水{fd['precipitation']}mm\n"
        chunks.append(chunk)

    for i in range(0, len(stations), 50):
        batch = stations[i:i+50]
        lines = [f"【{city_name}交通站点】第{i//50+1}批"]
        for s in batch:
            line = f"- {s['name']}：类型={s['type']}"
            if s["lines"]: line += f"，线路={s['lines']}"
            if s["operator"]: line += f"，运营={s['operator']}"
            lines.append(line)
        chunks.append("\n".join(lines))

    txt_path = os.path.join(DATA_DIR, f"{city_name}_chunks_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(chunks))
    print(f"[保存] RAG文本块 → {txt_path}（共{len(chunks)}块）")
    return chunks

def main():
    print("="*50)
    print(f"开始爬取 {TARGET_CITY} 城市数据")
    print("="*50)
    stations = fetch_transit_stations(TARGET_CITY)
    time.sleep(1)
    weather = fetch_weather(TARGET_CITY)
    chunks = save_and_chunk(stations, weather, TARGET_CITY)
    print(f"\n✅ 完成！站点:{len(stations)} 文本块:{len(chunks)}")
    print("下一步：python src/rag_engine.py")

if __name__ == "__main__":
    main()
