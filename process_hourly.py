import os
import json
import datetime
import math
import requests
import numpy as np
from scipy.interpolate import griddata
from PIL import Image

# 1. Pastikan folder output tersedia
os.makedirs("public/layers", exist_ok=True)

# 2. Bounding Box Wilayah Indonesia (Lat: -11 s/d 6, Lon: 95 s/d 141)
BBOX = {
    "south": -11.0,
    "north": 6.0,
    "west": 95.0,
    "east": 141.0
}

# Daftar stasiun/titik sampel representatif di Indonesia
STATIONS = [
    {"name": "Banda Aceh", "lat": 5.55, "lon": 95.32},
    {"name": "Medan", "lat": 3.59, "lon": 98.67},
    {"name": "Padang", "lat": -0.95, "lon": 100.35},
    {"name": "Palembang", "lat": -2.99, "lon": 104.76},
    {"name": "Jakarta", "lat": -6.20, "lon": 106.85},
    {"name": "Bandung", "lat": -6.91, "lon": 107.61},
    {"name": "Semarang", "lat": -6.97, "lon": 110.42},
    {"name": "Surabaya", "lat": -7.25, "lon": 112.75},
    {"name": "Denpasar", "lat": -8.67, "lon": 115.22},
    {"name": "Kupang", "lat": -10.17, "lon": 123.61},
    {"name": "Pontianak", "lat": -0.02, "lon": 109.34},
    {"name": "Banjarmasin", "lat": -3.32, "lon": 114.59},
    {"name": "Samarinda", "lat": -0.50, "lon": 117.15},
    {"name": "Manado", "lat": 1.48, "lon": 124.85},
    {"name": "Makassar", "lat": -5.15, "lon": 119.43},
    {"name": "Palu", "lat": -0.90, "lon": 119.87},
    {"name": "Ambon", "lat": -3.69, "lon": 128.18},
    {"name": "Ternate", "lat": 0.79, "lon": 127.38},
    {"name": "Sorong", "lat": -0.88, "lon": 131.25},
    {"name": "Jayapura", "lat": -2.53, "lon": 140.71},
    {"name": "Merauke", "lat": -8.49, "lon": 140.40}
]

def fetch_weather_data():
    """Mengambil suhu terkini secara real-time via Open-Meteo API"""
    lats = [s["lat"] for s in STATIONS]
    lons = [s["lon"] for s in STATIONS]
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={','.join(map(str, lats))}&longitude={','.join(map(str, lons))}&current=temperature_2m"
    
    print("[1/4] Mengambil data atmosfer real-time...")
    res = requests.get(url, timeout=20)
    data = res.json()
    
    # Jika multi-location, Open-Meteo mengembalikan list
    results = []
    if isinstance(data, list):
        for idx, item in enumerate(data):
            temp = item.get("current", {}).get("temperature_2m", 28.0)
            results.append((lons[idx], lats[idx], temp))
    else:
        temp = data.get("current", {}).get("temperature_2m", 28.0)
        results.append((lons[0], lats[0], temp))
        
    return results

def process_raster_and_manifest():
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    time_str = now_utc.strftime("%Y%m%d_%H00")
    label_wib = (now_utc + datetime.timedelta(hours=7)).strftime("%d %b %Y, %H:00 WIB")
    
    # 1. Ambil data
    points = fetch_weather_data()
    lons = np.array([p[0] for p in points])
    lats = np.array([p[1] for p in points])
    temps = np.array([p[2] for p in points])
    
    print(f"[2/4] Melakukan interpolasi spasial suhu untuk {label_wib}...")
    
    # 2. Buat grid resolusi reguler (Lebar: 460px, Tinggi: 170px)
    grid_w, grid_h = 460, 170
    grid_x = np.linspace(BBOX["west"], BBOX["east"], grid_w)
    grid_y = np.linspace(BBOX["north"], BBOX["south"], grid_h)
    gx, gy = np.meshgrid(grid_x, grid_y)
    
    # Interpolasi cubic / linear
    grid_temp = griddata((lons, lats), temps, (gx, gy), method='linear')
    # Isi area di luar convex hull dengan nearest neighbor
    grid_temp_nearest = griddata((lons, lats), temps, (gx, gy), method='nearest')
    grid_temp = np.where(np.isnan(grid_temp), grid_temp_nearest, grid_temp)
    
    # 3. Buat Gambar Overlay Termal (RGBA)
    print("[3/4] Menghasilkan raster visual...")
    img = Image.new("RGBA", (grid_w, grid_h))
    pixels = img.load()
    
    min_t, max_t = 20.0, 36.0 # Skala normalisasi (°C)
    for x in range(grid_w):
        for y in range(grid_h):
            t = grid_temp[y, x]
            norm = min(max((t - min_t) / (max_t - min_t), 0.0), 1.0)
            
            # Color ramp: Biru -> Cyan -> Kuning -> Merah
            r = int(255 * math.sin(norm * math.pi / 2))
            g = int(255 * math.sin(norm * math.pi))
            b = int(255 * math.cos(norm * math.pi / 2))
            a = 160  # Transparansi agar peta dasar terlihat
            pixels[x, y] = (r, g, b, a)
            
    output_filename = f"suhu_{time_str}.png"
    output_path = os.path.join("public/layers", output_filename)
    img.save(output_path, "PNG")
    
    # 4. Update manifest 'latest.json'
    manifest = {
        "status": "success",
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_label": label_wib,
        "layer_file": f"public/layers/{output_filename}",
        "bounds": [
            [BBOX["south"], BBOX["west"]],
            [BBOX["north"], BBOX["east"]]
        ],
        "stats": {
            "min_temp": round(float(np.min(temps)), 1),
            "max_temp": round(float(np.max(temps)), 1),
            "avg_temp": round(float(np.mean(temps)), 1)
        }
    }
    
    with open("public/latest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"[4/4] Sukses! File manifest diperbarui: public/latest.json")
    
    # 5. Bersihkan file lama (simpan hanya 24 jam terakhir agar repo hemat memori)
    clean_old_files(keep_hours=24)

def clean_old_files(keep_hours=24):
    """Menghapus raster yang lebih lama dari 24 jam"""
    layer_dir = "public/layers"
    if not os.path.exists(layer_dir):
        return
    files = sorted(os.listdir(layer_dir))
    if len(files) > keep_hours:
        for f in files[:-keep_hours]:
            try:
                os.remove(os.path.join(layer_dir, f))
                print(f"Pembersihan: menghapus file lama {f}")
            except Exception:
                pass

if __name__ == "__main__":
    process_raster_and_manifest()