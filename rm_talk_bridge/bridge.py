"""
RM Talk Bridge – Home Assistant add-on.

Live two-way audio between the RM Protect app and a UniFi Protect doorbell:
  phone mic  ──WebSocket (PCM 16 kHz)──▶ bridge ──ffmpeg (Opus/RTP)──▶ doorbell speaker
  doorbell mic (RTSPS, low latency) ──ffmpeg──▶ bridge ──WebSocket (PCM 16 kHz)──▶ phone

Protocol (wss://<host>:8086/talk):
  client → {"type":"auth","token":"<HA access token>"}
  server → {"type":"ready","sampleRate":16000}   or   {"type":"error","message":"…"}
  client → binary: mic audio, 16-bit little-endian mono PCM @ 16 kHz
  server → binary: doorbell audio, same format
  client → {"type":"stop"} (or just closes)
"""
import asyncio
import json
import logging
import ssl
import time
from pathlib import Path

import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("talk")

OPTIONS = json.loads(Path("/data/options.json").read_text())
PROTECT = f"https://{OPTIONS['protect_host']}/proxy/protect/integration/v1"
API_KEY = OPTIONS["protect_api_key"]
DOORBELL = OPTIONS["doorbell_id"]
HA_URL = OPTIONS["ha_url"].rstrip("/")
MAX_SECONDS = int(OPTIONS.get("max_session_seconds", 300))

SAMPLE_RATE = 16000
FRAME_BYTES = SAMPLE_RATE * 2 // 50          # 20 ms of 16-bit mono
IDLE_TIMEOUT = 6                             # seconds without mic audio → end
JITTER_TARGET = 3                            # frames (60 ms) buffered before sending to the doorbell
JITTER_MAX = 12                              # frames (240 ms): beyond this, drop the oldest to stay live

_busy = asyncio.Lock()                        # one talker at a time


# ── Helpers ──────────────────────────────────────────────────────────
async def ha_token_valid(session: aiohttp.ClientSession, token: str) -> bool:
    """The app's own HA login must be valid (checked against HA on the Pi)."""
    try:
        async with session.get(f"{HA_URL}/api/", headers={"Authorization": f"Bearer {token}"},
                               ssl=False, timeout=aiohttp.ClientTimeout(total=5)) as response:
            return response.status == 200
    except Exception as error:   # noqa: BLE001
        log.warning("HA check failed: %s", error)
        return False


async def protect(session: aiohttp.ClientSession, method: str, path: str) -> dict:
    async with session.request(method, f"{PROTECT}{path}", headers={"X-API-KEY": API_KEY, "Accept": "application/json"},
                               ssl=False, timeout=aiohttp.ClientTimeout(total=8)) as response:
        if response.status >= 300:
            raise RuntimeError(f"Protect {path} → HTTP {response.status}: {(await response.text())[:200]}")
        return await response.json(content_type=None)


async def rtsps_url(session: aiohttp.ClientSession) -> str:
    streams = await protect(session, "GET", f"/cameras/{DOORBELL}/rtsps-stream")
    url = streams.get("low") or streams.get("medium") or streams.get("high")
    if not url:
        raise RuntimeError("doorbell has no RTSPS stream")
    return url.split("?")[0]                  # ffmpeg doesn't want ?enableSrtp


def uplink_command(session: dict) -> list[str]:
    """Phone PCM → doorbell speaker, in the format Protect asked for."""
    codec = str(session.get("codec", "opus")).lower()
    rate = str(session.get("samplingRate", 24000))
    target = session["url"]                   # e.g. rtp://10.7.10.30:7004
    # Note: no "-fec" here: ffmpeg applies it to the RTP output (which rejects it), not the Opus encoder.
    encoder = {"opus": ["-c:a", "libopus", "-application", "voip", "-frame_duration", "20", "-b:a", "32k"],
               "aac": ["-c:a", "aac", "-b:a", "48k"],
               "pcmu": ["-c:a", "pcm_mulaw"], "pcma": ["-c:a", "pcm_alaw"]}.get(codec)
    if encoder is None:
        raise RuntimeError(f"unsupported talkback codec {codec}")
    return ["ffmpeg", "-nostdin", "-loglevel", "warning",
            "-fflags", "nobuffer", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
            *encoder, "-ar", rate, "-ac", "1",
            "-flush_packets", "1", "-f", "rtp", target]


def downlink_command(url: str) -> list[str]:
    """Doorbell mic (RTSPS) → plain PCM for the phone, with as little buffering as possible."""
    # Probe long enough to find the audio track (with probing at 0 ffmpeg can't see
    # the audio stream's parameters and exits at once), then run with no buffering.
    return ["ffmpeg", "-nostdin", "-loglevel", "warning",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "500000", "-analyzeduration", "1000000",
            "-rtsp_transport", "tcp", "-i", url,
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"]


async def stop(process: asyncio.subprocess.Process | None) -> None:
    if process and process.returncode is None:
        process.kill()
        await process.wait()


# ── WebSocket session ────────────────────────────────────────────────
async def handle_talk(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=10, max_msg_size=1 << 20)
    await ws.prepare(request)

    # 1. Authenticate with the app's HA token (first message, within 5 s)
    try:
        first = await ws.receive(timeout=5)
        token = json.loads(first.data).get("token", "") if first.type == aiohttp.WSMsgType.TEXT else ""
    except Exception:            # noqa: BLE001
        token = ""
    async with aiohttp.ClientSession() as http:
        if not token or not await ha_token_valid(http, token):
            await ws.send_json({"type": "error", "message": "Not signed in to Home Assistant"})
            await ws.close()
            return ws

        if _busy.locked():
            await ws.send_json({"type": "error", "message": "Someone else is already talking"})
            await ws.close()
            return ws

        async with _busy:
            uplink = downlink = None
            pump = pacer = None
            stderr_tasks = []
            try:
                # 2. Ask Protect for a talkback session and the doorbell's live audio
                session = await protect(http, "POST", f"/cameras/{DOORBELL}/talkback-session")
                listen_url = await rtsps_url(http)

                uplink = await asyncio.create_subprocess_exec(
                    *uplink_command(session), stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
                downlink = await asyncio.create_subprocess_exec(
                    *downlink_command(listen_url), stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)


                async def log_stderr(name, process):
                    while True:
                        line = await process.stderr.readline()
                        if not line:
                            break
                        log.warning("%s ffmpeg: %s", name, line.decode(errors="ignore").strip())
                stderr_tasks = [asyncio.create_task(log_stderr("uplink", uplink)),
                                asyncio.create_task(log_stderr("downlink", downlink))]

                async def pump_down():
                    while True:
                        chunk = await downlink.stdout.read(FRAME_BYTES)
                        if not chunk or ws.closed:
                            break
                        await ws.send_bytes(chunk)
                pump = asyncio.create_task(pump_down())

                # Jitter buffer: mic audio arrives in bursts over Wi-Fi/TCP; the doorbell needs a
                # steady 20 ms cadence. Buffer ~60 ms, then feed the encoder exactly in real time,
                # filling short gaps with silence and trimming if it falls behind.
                pending = bytearray()
                primed = asyncio.Event()

                async def pace_up():
                    await primed.wait()
                    silence = bytes(FRAME_BYTES)
                    next_tick = time.monotonic()
                    while uplink.returncode is None:
                        if len(pending) > JITTER_MAX * FRAME_BYTES:
                            del pending[: len(pending) - JITTER_TARGET * FRAME_BYTES]
                        if len(pending) >= FRAME_BYTES:
                            frame = bytes(pending[:FRAME_BYTES])
                            del pending[:FRAME_BYTES]
                        else:
                            frame = silence
                        uplink.stdin.write(frame)
                        await uplink.stdin.drain()
                        next_tick += 0.02
                        await asyncio.sleep(max(0, next_tick - time.monotonic()))
                pacer = asyncio.create_task(pace_up())

                await ws.send_json({"type": "ready", "sampleRate": SAMPLE_RATE})
                log.info("talk started")

                # 3. Mic audio in until stop / close / idle / max length
                started = time.monotonic()
                while time.monotonic() - started < MAX_SECONDS:
                    try:
                        message = await ws.receive(timeout=IDLE_TIMEOUT)
                    except asyncio.TimeoutError:
                        log.info("no mic audio for %ss, ending", IDLE_TIMEOUT)
                        break
                    if message.type == aiohttp.WSMsgType.BINARY:
                        if uplink.returncode is not None:
                            raise RuntimeError("talkback encoder stopped")
                        pending.extend(message.data)
                        if not primed.is_set() and len(pending) >= JITTER_TARGET * FRAME_BYTES:
                            primed.set()
                    elif message.type == aiohttp.WSMsgType.TEXT and json.loads(message.data).get("type") == "stop":
                        break
                    elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
            except Exception as error:   # noqa: BLE001
                log.warning("talk failed: %s", error)
                if not ws.closed:
                    await ws.send_json({"type": "error", "message": str(error)[:200]})
            finally:
                for task in (pump, pacer, *stderr_tasks):
                    if task:
                        task.cancel()
                await stop(uplink)
                await stop(downlink)
                if not ws.closed:
                    await ws.close()
                log.info("talk ended")
    return ws


async def handle_health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "busy": _busy.locked()})


def main() -> None:
    if not API_KEY or not DOORBELL:
        raise SystemExit("Set protect_api_key and doorbell_id in the add-on configuration")
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(f"/ssl/{OPTIONS['certfile']}", f"/ssl/{OPTIONS['keyfile']}")
    context.minimum_version = ssl.TLSVersion.TLSv1_2

    app = web.Application()
    # /talk on the LAN; /ui-talk-bridge/talk when reached through the Cloudflare tunnel path rule.
    app.add_routes([web.get("/talk", handle_talk), web.get("/ui-talk-bridge/talk", handle_talk),
                    web.get("/health", handle_health), web.get("/ui-talk-bridge/health", handle_health)])
    log.info("talk bridge on :8086 for doorbell %s via %s", DOORBELL, OPTIONS["protect_host"])
    web.run_app(app, host="0.0.0.0", port=8086, ssl_context=context, access_log=None)


if __name__ == "__main__":
    main()
