#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
py-xiaozhi WebSocket 版本
連接到本地 xiaozhi-esp32-server (WebSocket 協議)
"""
import json
import os
import time
import asyncio
import threading
import pyaudio
import opuslib
import websockets
from pynput import keyboard as pynput_keyboard
import serial

# ===================== 連線設定 =====================
WS_SERVER_URL = os.environ.get('XIAOZHI_WS_URL', 'ws://localhost:8000/xiaozhi/v1/')
DEVICE_ID = os.environ.get('XIAOZHI_DEVICE_ID', '')
CLIENT_ID = os.environ.get('XIAOZHI_CLIENT_ID', 'py-xiaozhi-ws-client')

# ===================== ESP32 眼球 Serial 設定 =====================
EYE_SERIAL_PORT = os.environ.get('EYE_SERIAL_PORT', 'COM10')
EYE_SERIAL_BAUD = 115200
eye_serial = None

EMOTION_TO_EYE = {
    'neutral':      '0',
    'happy':        '1',
    'angry':        '2',
    'sad':          '3',
    'surprised':    '4',
    'confused':     '5',
    'laughing':     '6',
    'funny':        '7',
    'loving':       '6',
    'winking':      '6',
    'kissy':        '6',
    'delicious':    '7',
    'silly':        '7',
    'crying':       '3',
    'embarrassed':  '9',
    'shocked':      '8',
    'thinking':     '5',
    'cool':         '6',
    'relaxed':      '8',
    'confident':    '7',
    'sleepy':       '9',
}
DEFAULT_EYE_INDEX = '0'
current_emotion = 'neutral'
emotion_reset_timer = None

# ===================== 全域狀態 =====================
session_id = None
key_state = "release"
listen_state = "stop"
tts_state = None
conn_state = False
ws_connection = None
audio = None
loop = None

# ===================== ESP32 眼球控制 =====================

def init_eye_serial():
    global eye_serial
    try:
        eye_serial = serial.Serial(EYE_SERIAL_PORT, EYE_SERIAL_BAUD, timeout=1)
        time.sleep(0.5)
        print(f"[EYE] Serial 連線成功: {EYE_SERIAL_PORT} @ {EYE_SERIAL_BAUD}")
    except Exception as e:
        print(f"[EYE] Serial 連線失敗 ({EYE_SERIAL_PORT}): {e}")
        print("[EYE] 眼球控制功能將被停用，程式繼續運行")
        eye_serial = None


def send_eye_command(design_index_char):
    global eye_serial
    if eye_serial and eye_serial.is_open:
        try:
            eye_serial.write(design_index_char.encode('ascii'))
            eye_serial.flush()
            print(f"[EYE] 傳送眼球指令: '{design_index_char}'")
        except Exception as e:
            print(f"[EYE] 傳送指令失敗: {e}")


def set_emotion_eye(emotion_str):
    global current_emotion, emotion_reset_timer
    if emotion_str is None:
        return
    emotion_lower = emotion_str.lower().strip()
    if emotion_lower == current_emotion:
        return
    eye_idx = EMOTION_TO_EYE.get(emotion_lower, DEFAULT_EYE_INDEX)
    current_emotion = emotion_lower
    print(f"[EYE] 情緒變化: {emotion_lower} → 眼球設計 #{eye_idx}")
    send_eye_command(eye_idx)
    if emotion_reset_timer:
        emotion_reset_timer.cancel()
        emotion_reset_timer = None


def reset_emotion_eye(delay=5.0):
    global emotion_reset_timer
    if emotion_reset_timer:
        emotion_reset_timer.cancel()
    emotion_reset_timer = threading.Timer(delay, _do_reset_emotion)
    emotion_reset_timer.daemon = True
    emotion_reset_timer.start()


def _do_reset_emotion():
    global current_emotion
    if current_emotion != 'neutral':
        print("[EYE] TTS 結束，恢復中性眼球")
        current_emotion = 'neutral'
        send_eye_command(DEFAULT_EYE_INDEX)


# ===================== WebSocket 訊息發送 =====================

def send_ws_text(message):
    """透過 WebSocket 發送 JSON 文字訊息"""
    global ws_connection, loop
    if ws_connection and loop:
        asyncio.run_coroutine_threadsafe(
            ws_connection.send(json.dumps(message)), loop
        )


def send_ws_bytes(data):
    """透過 WebSocket 發送二進位音訊資料"""
    global ws_connection, loop
    if ws_connection and loop:
        asyncio.run_coroutine_threadsafe(
            ws_connection.send(data), loop
        )


# ===================== 音訊發送（麥克風 → Server）=====================

def send_audio_thread_func():
    """持續從麥克風讀取音訊，Opus 編碼後透過 WebSocket 發送"""
    global listen_state, audio, ws_connection
    encoder = opuslib.Encoder(16000, 1, opuslib.APPLICATION_AUDIO)
    mic = audio.open(format=pyaudio.paInt16, channels=1, rate=16000,
                     input=True, frames_per_buffer=960)
    try:
        while True:
            if listen_state == "stop":
                time.sleep(0.05)
                continue
            data = mic.read(960, exception_on_overflow=False)
            encoded_data = encoder.encode(data, 960)
            if ws_connection:
                send_ws_bytes(bytes(encoded_data))
    except Exception as e:
        print(f"[Audio Send] 錯誤: {e}")
    finally:
        mic.stop_stream()
        mic.close()


# ===================== 音訊接收（Server → 喇叭）=====================

audio_play_queue = asyncio.Queue()


async def play_audio_worker():
    """從佇列中取出音訊資料並播放"""
    global audio
    sample_rate = 24000
    frame_duration = 60
    frame_num = int(sample_rate * frame_duration / 1000)
    decoder = opuslib.Decoder(sample_rate, 1)
    spk = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                     output=True, frames_per_buffer=frame_num)
    try:
        while True:
            data = await audio_play_queue.get()
            if data is None:
                break
            try:
                pcm = decoder.decode(data, frame_num)
                spk.write(pcm)
            except Exception as e:
                print(f"[Audio Play] 解碼錯誤: {e}")
    finally:
        spk.stop_stream()
        spk.close()


# ===================== WebSocket 訊息處理 =====================

async def on_ws_message(message):
    """處理從 Server 收到的訊息"""
    global session_id, tts_state

    if isinstance(message, bytes):
        await audio_play_queue.put(message)
        return

    msg = json.loads(message)
    print(f"[WS Recv] {msg}")

    if msg['type'] == 'hello':
        session_id = msg.get('session_id')
        print(f"[WS] 連線建立，session_id: {session_id}")

    if 'emotion' in msg:
        set_emotion_eye(msg['emotion'])

    if msg['type'] == 'tts':
        tts_state = msg.get('state')
        if tts_state == 'stop':
            print("[TTS] 播放結束，按住空白鍵繼續對話")
            reset_emotion_eye(delay=5.0)

    if msg['type'] == 'llm':
        if 'emotion' in msg:
            set_emotion_eye(msg['emotion'])

    if msg['type'] == 'goodbye':
        if msg.get('session_id') == session_id:
            print("[WS] 收到 goodbye")
            reset_emotion_eye(delay=0)
            session_id = None


# ===================== 鍵盤控制 =====================

def on_space_key_press(event):
    global key_state, listen_state, session_id, conn_state, tts_state
    if key_state == "press":
        return
    key_state = "press"

    if conn_state is False or session_id is None:
        conn_state = True
        hello_msg = {
            "type": "hello", "version": 3, "transport": "websocket",
            "audio_params": {"format": "opus", "sample_rate": 16000,
                             "channels": 1, "frame_duration": 60}
        }
        send_ws_text(hello_msg)
        print(f"[WS Send] hello: {hello_msg}")
    else:
        if tts_state == "start" or tts_state == "sentence_start":
            send_ws_text({"type": "abort"})
            print("[WS Send] abort")
        msg = {"session_id": session_id, "type": "listen",
               "state": "start", "mode": "manual"}
        print(f"[WS Send] listen start: {msg}")
        send_ws_text(msg)

    listen_state = "start"


def on_space_key_release(event):
    global session_id, key_state, listen_state
    key_state = "release"
    listen_state = "stop"
    if session_id is not None:
        msg = {"session_id": session_id, "type": "listen", "state": "stop"}
        print(f"[WS Send] listen stop: {msg}")
        send_ws_text(msg)


def on_press(key):
    if key == pynput_keyboard.Key.space:
        on_space_key_press(None)
    elif hasattr(key, 'char') and key.char and key.char.isdigit():
        digit = key.char
        if '0' <= digit <= '9':
            print(f"[EYE] 手動切換眼球設計: #{digit}")
            send_eye_command(digit)
    elif hasattr(key, 'char') and key.char:
        emotion_shortcuts = {
            'h': 'happy', 'a': 'angry', 's': 'sad',
            'u': 'surprised', 'c': 'confused', 'n': 'neutral',
            'l': 'loving', 'k': 'funny', 'd': 'shocked', 'z': 'sleepy',
        }
        if key.char in emotion_shortcuts:
            set_emotion_eye(emotion_shortcuts[key.char])


def on_release(key):
    if key == pynput_keyboard.Key.space:
        on_space_key_release(None)
    if key == pynput_keyboard.Key.esc:
        return False


# ===================== 主程式 =====================

async def ws_main():
    global ws_connection, loop
    loop = asyncio.get_running_loop()

    headers = {
        'device-id': DEVICE_ID,
        'client-id': CLIENT_ID,
    }

    print(f"[WS] 正在連接 {WS_SERVER_URL} ...")

    # websockets >=14 renamed extra_headers → additional_headers
    ws_kwargs = dict(ping_interval=30, ping_timeout=120)
    if hasattr(websockets, 'version') and int(websockets.version.split('.')[0]) >= 14:
        ws_kwargs['additional_headers'] = headers
    else:
        ws_kwargs['extra_headers'] = headers

    async with websockets.connect(WS_SERVER_URL, **ws_kwargs) as ws:
        ws_connection = ws
        print("[WS] WebSocket 連線成功！")
        print("[操作說明] 按住空白鍵說話，放開停止；ESC 退出")

        audio_task = asyncio.create_task(play_audio_worker())

        mic_thread = threading.Thread(target=send_audio_thread_func, daemon=True)
        mic_thread.start()

        try:
            async for message in ws:
                await on_ws_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[WS] 連線關閉: {e}")
        finally:
            await audio_play_queue.put(None)
            await audio_task
            ws_connection = None


def run():
    global audio
    audio = pyaudio.PyAudio()
    init_eye_serial()

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("=" * 60)
    print("  py-xiaozhi WebSocket 版本")
    print(f"  Server: {WS_SERVER_URL}")
    print(f"  Device ID: {DEVICE_ID}")
    print("=" * 60)

    try:
        asyncio.run(ws_main())
    except KeyboardInterrupt:
        print("\n[Exit] 使用者中斷")
    finally:
        audio.terminate()


if __name__ == "__main__":
    run()
