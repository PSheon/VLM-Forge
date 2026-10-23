"""Grab one frame per NVR channel over RTSP: the detect line's training frames (GOALS §7).

    forge capture --out workspace/captures/$(date +%Y%m%d-%H%M) \
        --url 'rtsp://USER:PASS@{ip}:554/...{ch}...' --nvr 10.12.31.6 10.12.31.8 --channels 1-16

Writes ``<out>/nvr_<ip with _>/nvr<last octet>_ch<NN>.jpg`` (the held-out source's naming) and
``capture.json`` (url template with the password masked, time per frame, failures). Register
the directory in the UI with "掃描目錄" (source ``capture:<dir name>``) and build a queue on it.
Needs the optional ``capture`` extra (PyAV).
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path


def parse_channels(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def mask(url: str) -> str:
    return re.sub(r"//([^:/@]+):([^@]*)@", r"//\1:***@", url)


def grab(url: str, timeout_s: float = 8.0) -> bytes:
    """First decodable frame of an RTSP stream as JPEG bytes."""
    import io

    import av

    opts = {"rtsp_transport": "tcp", "stimeout": str(int(timeout_s * 1e6))}
    with av.open(url, options=opts, timeout=timeout_s) as c:
        stream = c.streams.video[0]
        for frame in c.decode(stream):
            buf = io.BytesIO()
            frame.to_image().save(buf, format="JPEG", quality=95)
            return buf.getvalue()
    raise RuntimeError("no frame")


def capture(url_template: str, nvrs: list[str], channels: list[int], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log = {
        "url": mask(url_template),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "frames": [],
        "failed": [],
    }
    for ip in nvrs:
        d = out / f"nvr_{ip.replace('.', '_')}"
        d.mkdir(exist_ok=True)
        last = ip.split(".")[-1]
        for ch in channels:
            url = url_template.format(ip=ip, ch=ch, ch2=f"{ch:02d}")
            path = d / f"nvr{last}_ch{ch:02d}.jpg"
            t0 = time.time()
            try:
                path.write_bytes(grab(url))
                log["frames"].append({"path": str(path), "s": round(time.time() - t0, 2)})
                print(f"ok   {path}", flush=True)
            except Exception as e:  # noqa: BLE001 - one dead channel must not stop the sweep
                log["failed"].append({"nvr": ip, "ch": ch, "error": f"{type(e).__name__}: {e}"})
                print(f"fail {ip} ch{ch:02d}: {e}", flush=True)
    (out / "capture.json").write_text(json.dumps(log, indent=1))
    return log
