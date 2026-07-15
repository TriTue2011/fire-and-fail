# Chạy ổn định RTSP @ 10 hoặc 15 FPS

## Kết luận từ GitHub / cộng đồng

| Nguồn | Khuyến nghị |
|--------|-------------|
| **Intel Geti** `IPCameraStream` | `rtsp_transport;tcp` + `fflags;nobuffer` + `flags;low_delay` + `stimeout` |
| **OpenCV forum / SO** | `CAP_PROP_BUFFERSIZE=1`, reader thread chỉ giữ **frame mới nhất** |
| **YOLO + RTSP blogs** | Không inference 25–30fps trên CPU multi-cam; throttle 10–15fps |
| **Camera vendors** | Dùng **substream** (`subtype=1` / `Channels/102`) cho AI, mainstream cho NVR |

## Cách bật 15 FPS (mặc định)

```bash
# Windows PowerShell
$env:SMARTSENSE_TARGET_FPS="15"
python "Fall detection.py"
```

## Cách bật 10 FPS (máy yếu / ≥3 camera)

```bash
$env:SMARTSENSE_TARGET_FPS="10"
python "Fall detection.py"
```

Hoặc trong `cameras.json`:

```json
{
  "target_fps": 10,
  "cameras": {
    "Cam1": "rtsp://admin:pass@192.168.1.3:554/cam/realmonitor?channel=1&subtype=1"
  }
}
```

## URL RTSP gợi ý (substream = nhẹ)

| Hãng | URL |
|------|-----|
| Dahua / Kbvision / KBONE | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=1` |
| Hikvision / Hilook | `rtsp://user:pass@IP:554/Streaming/Channels/102` |
| EZVIZ | `rtsp://admin:verify_code@IP:554/H.264` |
| Yoosee | `rtsp://IP/onvif1` |

`subtype=0` / `101` = mainstream (thường 1080p–4K 25fps) → dễ lag CPU.  
`subtype=1` / `102` = substream (thường D1/720p 10–15fps) → **nên dùng cho AI**.

## Kiểm tra health

Mở trình duyệt: `http://localhost:9000/health`

```json
{
  "target_fps": 15,
  "cameras": {
    "Cam1": {
      "rtsp_fps": 14.8,
      "infer_fps": 14.2,
      "connected": true
    }
  }
}
```

- `rtsp_fps` ~ camera decode rate  
- `infer_fps` ~ số lần YOLO/giây (nên ≤ `target_fps`)  
- Nếu `last_frame_age_s` > 3 → RTSP đứt / sai URL / firewall

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `SMARTSENSE_TARGET_FPS` | `15` | Chỉ `10` hoặc `15` |
| `SMARTSENSE_IMG_SIZE` | `320` | `320` nhanh, `416`/`640` chính xác hơn |
| `SMARTSENSE_CONF` | `0.35` | Ngưỡng YOLO pose |
| `TELEGRAM_BOT_TOKEN` | — | Token bot |
| `TELEGRAM_CHAT_ID` | — | Chat nhận cảnh báo |

## Tối ưu thêm nếu vẫn chậm

1. Giảm camera song song (1–2 cam / CPU 4 lõi).  
2. `SMARTSENSE_IMG_SIZE=320`, `TARGET_FPS=10`.  
3. Trên camera: set encode **H.264**, bitrate thấp, **Capped VBR**.  
4. GPU NVIDIA: cài CUDA + `ultralytics` tự dùng GPU.  
5. Tránh mở nhiều client xem live cùng lúc (MJPEG tốn băng thông).
