import time
from pathlib import Path
import numpy as np
from xlib import os as lib_os
from xlib.mp import csw as lib_csw

# -------- 修复：异常语句行残留标记 & 安全导入 sounddevice --------
try:
    import sounddevice as sd
except Exception:
    sd = None
import threading, queue, platform

# -------- 新增：bc_out 输出独占，避免并发写导致闪烁 --------
_BC_OUT_LOCK = threading.Lock()
_BC_OUT_OWNER = {}

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)

# NOTE: This panel is now AUDIO SYNC ONLY.
# Face wrap / face swap functionality has been fully removed for this UI to avoid any GPU/CPU overhead.
# Frames are passed through unchanged; only microphone->speaker delay is handled here.

class FaceSoundPicture(BackendHost):
    def __init__(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, faces_path : Path, backend_db : BackendDB = None,
                  id : int = 0):
        self._id = id
        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=FaceSoundPictureWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out, faces_path])

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()

    def _get_name(self):
        return super()._get_name()

class FaceSoundPictureWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, faces_path : Path):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out
        self.faces_path = faces_path

        # 新增：bc_out 所有权标记
        self._owns_bc_out = False

        self.pending_bcd = None

        lib_os.set_timer_resolution(1)

        state, cs = self.get_state(), self.get_control_sheet()

        # === AV Sync bindings ===
        cs.mic_device.call_on_selected(self.on_cs_mic)
        cs.spk_device.call_on_selected(self.on_cs_spk)
        cs.av_delay_ms.call_on_number(self.on_cs_delay)
        cs.av_delay_enable.call_on_selected(self.on_cs_av_delay_enable)

        # 绑定刷新设备列表信号
        if hasattr(cs, 'refresh_devices'):
            cs.refresh_devices.call_on_signal(self._on_refresh_devices)
            cs.refresh_devices.enable()

        # 初始化并填充设备列表（仅在 Windows 显示真实列表；其他平台仅给默认项）
        self._populate_audio_devices()

        # 初始化延迟开关（默认关闭）
        cs.av_delay_enable.enable()
        cs.av_delay_enable.set_choices(['关闭','开启'])
        if state.av_delay_enable:
            cs.av_delay_enable.select('开启')
        else:
            cs.av_delay_enable.select('关闭')

        # 如果持久化状态为已开启，则尝试在启动时占用输出通道
        if state.av_delay_enable and not getattr(self, '_owns_bc_out', False):
            self._claim_bc_out()

        # 设置默认延迟 2500ms
        cs.av_delay_ms.enable()
        cs.av_delay_ms.set_config(lib_csw.Number.Config(min=0, max=10000, step=10, decimals=0, allow_instant_update=True))
        default_delay = 2500 if state.av_delay_ms is None else int(state.av_delay_ms)
        cs.av_delay_ms.set_number(default_delay)
        state.av_delay_ms = default_delay

        # 启动/重建音频延迟引擎（开关关闭或设备未选时会直接返回）
        self._rebuild_engine()

    # -------- 新增：占用/释放 bc_out 的帮助方法 --------
    def _claim_bc_out(self):
        key = id(self.bc_out)
        with _BC_OUT_LOCK:
            owner = _BC_OUT_OWNER.get(key)
            if owner is None or owner is self:
                _BC_OUT_OWNER[key] = self
                self._owns_bc_out = True
                return True
            return False

    def _release_bc_out(self):
        key = id(self.bc_out)
        with _BC_OUT_LOCK:
            owner = _BC_OUT_OWNER.get(key)
            if owner is self:
                _BC_OUT_OWNER.pop(key, None)
        self._owns_bc_out = False

    def _populate_audio_devices(self):
        state, cs = self.get_state(), self.get_control_sheet()
        names_in, names_out, def_in, def_out = self._enum_audio_devices()

        # Mic selector
        cs.mic_device.enable()
        cs.mic_device.set_choices(names_in or [], none_choice_name='@misc.menu_select')
        # Priority: persisted state -> system default -> unselect
        mic_sel = state.mic_device if (state.mic_device in (names_in or [])) else (def_in if def_in in (names_in or []) else None)
        if mic_sel is not None:
            cs.mic_device.select(mic_sel)
        else:
            cs.mic_device.unselect()
        state.mic_device = mic_sel

        # Speaker selector
        cs.spk_device.enable()
        cs.spk_device.set_choices(names_out or [], none_choice_name='@misc.menu_select')
        spk_sel = state.spk_device if (state.spk_device in (names_out or [])) else (def_out if def_out in (names_out or []) else None)
        if spk_sel is not None:
            cs.spk_device.select(spk_sel)
        else:
            cs.spk_device.unselect()
        state.spk_device = spk_sel

    def _enum_audio_devices(self):
        """Returns (input_names, output_names, default_input_name, default_output_name)
        - Names are raw device names (no host API suffix)
        - Defaults are raw names or None if unknown
        """
        if platform.system().lower() != 'windows' or sd is None:
            return [], [], None, None
        try:
            devices = sd.query_devices()
            inputs, outputs = [], []
            for d in devices:
                name = d.get('name', '')
                if not name:
                    continue
                if d.get('max_input_channels', 0) > 0 and name not in inputs:
                    inputs.append(name)
                if d.get('max_output_channels', 0) > 0 and name not in outputs:
                    outputs.append(name)
            # Resolve default names from indices, if provided
            def_in = def_out = None
            try:
                di, do = sd.default.device
                if isinstance(di, int) and 0 <= di < len(devices):
                    def_in = devices[di].get('name') or None
                if isinstance(do, int) and 0 <= do < len(devices):
                    def_out = devices[do].get('name') or None
            except Exception:
                pass
            return inputs, outputs, def_in, def_out
        except Exception:
            return [], [], None, None

    def _rebuild_engine(self):
        # Stop existing
        try:
            if hasattr(self, '_audio_engine') and self._audio_engine is not None:
                self._audio_engine.stop(); self._audio_engine = None
        except Exception:
            self._audio_engine = None
        # Build new only when devices are selected AND delay effect is enabled
        st = self.get_state()
        if not getattr(st, 'av_delay_enable', False):
            return
        if st.mic_device is None or st.spk_device is None:
            return
        # 新增：必须已经占有 bc_out 才能创建引擎
        if not getattr(self, '_owns_bc_out', False):
            return
        self._audio_engine = AudioDelayEngine(self)
        self._audio_engine.start()

    def _norm_sel(self, v):
        # 将占位项或空值标准化为 None
        if v is None:
            return None
        if isinstance(v, str) and v.startswith('@'):
            return None
        return v

    def on_cs_mic(self, idx, name):
        st = self.get_state()
        st.mic_device = self._norm_sel(name)
        self.save_state()
        if st.mic_device is not None and st.spk_device is not None and getattr(st, 'av_delay_enable', False) and getattr(self, '_owns_bc_out', False):
            self._rebuild_engine()
        else:
            if hasattr(self, '_audio_engine') and self._audio_engine is not None:
                try:
                    self._audio_engine.stop()
                except Exception:
                    pass
                self._audio_engine = None
        self.reemit_frame_signal.send()

    def on_cs_spk(self, idx, name):
        st = self.get_state()
        st.spk_device = self._norm_sel(name)
        self.save_state()
        if st.mic_device is not None and st.spk_device is not None and getattr(st, 'av_delay_enable', False) and getattr(self, '_owns_bc_out', False):
            self._rebuild_engine()
        else:
            if hasattr(self, '_audio_engine') and self._audio_engine is not None:
                try:
                    self._audio_engine.stop()
                except Exception:
                    pass
                self._audio_engine = None
        self.reemit_frame_signal.send()

    def on_cs_delay(self, v):
        st, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.av_delay_ms.get_config()
        st.av_delay_ms = int(np.clip(v, cfg.min, cfg.max)); cs.av_delay_ms.set_number(st.av_delay_ms)
        self.save_state();
        if hasattr(self, '_audio_engine') and self._audio_engine is not None:
            self._audio_engine.set_delay_ms(st.av_delay_ms)
        self.reemit_frame_signal.send()

    # -------- 替换：开启前先抢占 bc_out；占不到则自动回退为关闭 --------
    def on_cs_av_delay_enable(self, idx, name):
        st = self.get_state()
        cs = self.get_control_sheet()
        # 将选择名称映射为布尔开关
        on_names = {'开启', 'On', 'ON', 'on', '1', 1, True}
        st.av_delay_enable = (name in on_names)
        self.save_state()
        # 根据开关启动/停止引擎；开启时先尝试占用输出，关闭时释放
        if st.av_delay_enable:
            if not getattr(self, '_owns_bc_out', False):
                if not self._claim_bc_out():
                    # 无法占用输出：回退为关闭，避免与其他换脸模块并发写导致闪烁
                    st.av_delay_enable = False
                    try:
                        cs.av_delay_enable.select('关闭')
                    except Exception:
                        pass
                    if hasattr(self, '_audio_engine') and self._audio_engine is not None:
                        try:
                            self._audio_engine.set_delay_ms(0)
                            self._audio_engine.stop()
                        except Exception:
                            pass
                        self._audio_engine = None
                    self.save_state(); self.reemit_frame_signal.send(); return
            # 拿到所有权，启动/更新引擎
            if getattr(self, '_audio_engine', None) is None:
                self._rebuild_engine()
            else:
                try:
                    self._audio_engine.set_delay_ms(int(st.av_delay_ms))
                except Exception:
                    pass
        else:
            # 关闭：停止引擎并释放所有权
            if hasattr(self, '_audio_engine') and self._audio_engine is not None:
                try:
                    self._audio_engine.set_delay_ms(0)
                    self._audio_engine.stop()
                except Exception:
                    pass
                self._audio_engine = None
            self._release_bc_out()
        self.reemit_frame_signal.send()

    # -------- 替换：设备刷新时也要求“已开启 + 已占有”才重建，否则停机 --------
    def _on_refresh_devices(self):
        # 重新枚举并尽量保持/回填选择
        self._populate_audio_devices()
        st = self.get_state()
        if st.mic_device is not None and st.spk_device is not None and getattr(st, 'av_delay_enable', False) and getattr(self, '_owns_bc_out', False):
            self._rebuild_engine()
        else:
            if hasattr(self, '_audio_engine') and self._audio_engine is not None:
                try:
                    self._audio_engine.stop()
                except Exception:
                    pass
                self._audio_engine = None
        self.reemit_frame_signal.send()

    # -------- 新增：停止时释放引擎与所有权 --------
    def on_stop(self):
        try:
            if hasattr(self, '_audio_engine') and self._audio_engine is not None:
                self._audio_engine.stop()
        except Exception:
            pass
        self._audio_engine = None
        self._release_bc_out()

    def on_tick(self):
        # Audio-only module: do not participate in video pipeline to avoid conflicts with FaceSwap/DFM
        return

    def _resolve_device_index(self, name, want_output=False):
        if name is None:
            return None
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d.get('name') == name:
                    if want_output and d.get('max_output_channels',0) > 0:
                        return i
                    if not want_output and d.get('max_input_channels',0) > 0:
                        return i
        except Exception:
            pass
        # fallback to system default index
        try:
            di, do = sd.default.device
            return (do if want_output else di)
        except Exception:
            return None

class AudioDelayEngine:
    def __init__(self, worker: 'FaceSoundPictureWorker'):
        self.worker = worker
        self.thread = None
        self.stop_evt = threading.Event()
        self.q = queue.Queue(maxsize=64)
        self.samplerate = 48000
        self.channels_in = 1
        self.channels_out = 2
        self.buffered_frames = 0
        self.started = False

    def start(self):
        if sd is None or platform.system().lower() != 'windows':
            return
        st = self.worker.get_state()
        self.delay_ms = int(st.av_delay_ms) if getattr(st, 'av_delay_enable', False) else 0
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.stop_evt.clear()
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        try:
            if self.thread: self.thread.join(timeout=1.0)
        except Exception:
            pass
        # 建议：置空线程句柄，便于多次启停
        self.thread = None

    def set_delay_ms(self, ms:int):
        prev = getattr(self, 'delay_ms', 0)
        self.delay_ms = int(ms)
        # If delay increased, require rebuffering to realize the larger delay
        if self.delay_ms > prev:
            self.started = False
        # If delay set to zero, flush buffer and switch to immediate passthrough
        if self.delay_ms == 0:
            try:
                while not self.q.empty():
                    _ = self.q.get_nowait()
            except Exception:
                pass
            self.buffered_frames = 0
            self.started = True

    def _resolve_device_index(self, name, want_output=False):
        if name is None:
            return None
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d.get('name') == name:
                    if want_output and d.get('max_output_channels',0) > 0:
                        return i
                    if not want_output and d.get('max_input_channels',0) > 0:
                        return i
        except Exception:
            pass
        # fallback to system default index
        try:
            di, do = sd.default.device
            return (do if want_output else di)
        except Exception:
            return None

    def _run(self):
        try:
            st = self.worker.get_state()
            mic_idx = self._resolve_device_index(st.mic_device, want_output=False)
            spk_idx = self._resolve_device_index(st.spk_device, want_output=True)
            if mic_idx is None or spk_idx is None:
                return

            # Query rates
            try:
                dinfo = sd.query_devices(mic_idx)
                self.samplerate = int(dinfo.get('default_samplerate', self.samplerate))
            except Exception:
                pass

            blocksize = max(256, int(self.samplerate * 0.02))  # ~20ms
            delay_frames = int(self.samplerate * (self.delay_ms/1000.0))
            # ensure buffer can hold delay
            self.q = queue.Queue(maxsize=max(64, delay_frames//blocksize + 8))
            self.buffered_frames = 0
            self.started = False

            def in_cb(indata, frames, time_info, status):
                if status: pass
                # mono mixdown
                if indata.ndim > 1:
                    mono = np.mean(indata, axis=1, keepdims=True)
                else:
                    mono = indata.reshape(-1,1)
                try:
                    self.q.put_nowait(mono.copy())
                    self.buffered_frames += len(mono)
                except queue.Full:
                    try:
                        _ = self.q.get_nowait()
                        self.q.put_nowait(mono.copy())
                        # do not change self.buffered_frames here: dropping oldest then adding same length keeps total buffered the same
                    except Exception:
                        pass
                return (None, sd.CallbackFlags())

            def out_cb(outdata, frames, time_info, status):
                if status: pass
                buf = np.zeros((frames, 1), dtype=np.float32)

                # Recompute current target delay in frames each callback to reflect runtime changes
                current_delay_frames = int(self.samplerate * (self.delay_ms/1000.0))

                # If not started, check buffer fill level; output silence until enough is buffered
                if not self.started:
                    if self.buffered_frames >= current_delay_frames:
                        self.started = True
                    else:
                        out = np.repeat(buf, 2, axis=1)
                        outdata[:] = out
                        return (None, sd.CallbackFlags())

                # Normal delayed playback: consume from queue
                i = 0
                while i < frames:
                    if self.q.empty():
                        break
                    chunk = self.q.get()
                    take = min(len(chunk), frames - i)
                    buf[i:i+take, 0] = chunk[:take, 0]
                    # update buffered_frames accounting
                    self.buffered_frames = max(0, self.buffered_frames - take)
                    if take < len(chunk):
                        rest = chunk[take:]
                        try:
                            self.q.put_nowait(rest)
                        except Exception:
                            pass
                    i += take

                out = np.repeat(buf, 2, axis=1)
                outdata[:] = out
                return (None, sd.CallbackFlags())

            with sd.InputStream(device=mic_idx, channels=1, samplerate=self.samplerate, blocksize=blocksize, callback=in_cb):
                with sd.OutputStream(device=spk_idx, channels=2, samplerate=self.samplerate, blocksize=blocksize, callback=out_cb):
                    while not self.stop_evt.is_set():
                        time.sleep(0.01)
        except Exception:
            pass

class Sheet:
    class Host(lib_csw.Sheet.Host):
        def __init__(self):
            super().__init__()
            self.mic_device  = lib_csw.DynamicSingleSwitch.Client()
            self.spk_device  = lib_csw.DynamicSingleSwitch.Client()
            self.av_delay_enable = lib_csw.DynamicSingleSwitch.Client()
            self.av_delay_ms = lib_csw.Number.Client()
            self.refresh_devices = lib_csw.Signal.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.mic_device  = lib_csw.DynamicSingleSwitch.Host()
            self.spk_device  = lib_csw.DynamicSingleSwitch.Host()
            self.av_delay_enable = lib_csw.DynamicSingleSwitch.Host()
            self.av_delay_ms = lib_csw.Number.Host()
            self.refresh_devices = lib_csw.Signal.Host()

class WorkerState(BackendWorkerState):
    mic_device : str = None
    spk_device : str = None
    av_delay_ms : int = 2500
    av_delay_enable : bool = False
