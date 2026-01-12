import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
from modelhub import DFLive
from xlib import os as lib_os
from xlib.image.ImageProcessor import ImageProcessor
from xlib.mp import csw as lib_csw
from xlib.python import all_is_not_None
import os
import sys

# 2026 兼容性补丁：解决 CuPy 找不到 nvrtc64_130_0.dll 的问题
cuda_path = os.environ.get('CUDA_PATH') # 默认通常是 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
if cuda_path:
    cuda_bin = os.path.join(cuda_path, 'bin')
    if os.path.exists(cuda_bin):
        if sys.version_info >= (3, 8):
            os.add_dll_directory(cuda_bin)
        else:
            os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']


# 导入补全的识别模块
from modelhub.onnx import InsightFaceRecognition

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)

class FaceSwapDFM(BackendHost):
    def __init__(self, weak_heap : BackendWeakHeap, reemit_frame_signal : BackendSignal, bc_in : BackendConnection, bc_out : BackendConnection, dfm_models_path : Path, backend_db : BackendDB = None,
                  id : int = 0):
        self._id = id
        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=FaceSwapDFMWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out, dfm_models_path])

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()

class FaceSwapDFMWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap, reemit_frame_signal, bc_in, bc_out, dfm_models_path):
        self.weak_heap, self.reemit_frame_signal = weak_heap, reemit_frame_signal
        self.bc_in, self.bc_out, self.dfm_models_path = bc_in, bc_out, dfm_models_path
        
        self.pending_bcd = None
        self.dfm_model_initializer = None
        self.dfm_model = None

        # --- 2026 核心锁定变量初始化 ---
        self.rec_model = None
        self.known_embeddings = []
        self.auto_capture_mode = True  # 初始设为 True，实现启动即自动捕获
        self.rec_initializing = False

        lib_os.set_timer_resolution(1)
        state, cs = self.get_state(), self.get_control_sheet()

        # 注册 UI 回调
        cs.model.call_on_selected(self.on_cs_model)
        cs.device.call_on_selected(self.on_cs_device)
        cs.swap_all_faces.call_on_flag(self.on_cs_swap_all_faces)
        cs.capture_flag.call_on_flag(self.on_cs_capture_flag)
        cs.face_id.call_on_number(self.on_cs_face_id)
        cs.morph_factor.call_on_number(self.on_cs_morph_factor)
        cs.presharpen_amount.call_on_number(self.on_cs_presharpen_amount)
        
        # 动态绑定 Gamma 回调
        for attr in ['pre_gamma_red', 'pre_gamma_green', 'pre_gamma_blue', 'post_gamma_red', 'post_gamma_green', 'post_gamma_blue']:
            # 确保 getattr 能够找到对应的方法名，如 self.on_cs_pre_gamma_red
            func = getattr(self, f'on_cs_{attr}', None)
            if func:
                getattr(cs, attr).call_on_number(func)
        
        cs.two_pass.call_on_flag(self.on_cs_two_pass)

        # 启动时初始化设备列表
        cs.device.enable()
        cs.device.set_choices(DFLive.get_available_devices(), none_choice_name='@misc.menu_select')
        
        if state.device is not None:
            cs.device.select(state.device)
            # 2026 修复：设备选定后，立即刷新可用模型列表
            models_info = DFLive.get_available_models_info(self.dfm_models_path)
            cs.model.enable()
            cs.model.set_choices(models_info, none_choice_name='@misc.menu_select')
            if state.model is not None:
                cs.model.select(state.model)
        
        # 激活配置下发
        self._update_ui_configs()
            # 核心修复：补全 UI 交互函数
    def on_cs_swap_all_faces(self, v): self._update_model_state('swap_all_faces', v)
    def on_cs_face_id(self, v): self._update_model_state('face_id', int(v))
    def on_cs_morph_factor(self, v): self._update_model_state('morph_factor', v)
    def on_cs_presharpen_amount(self, v): self._update_model_state('presharpen_amount', v)
    def on_cs_pre_gamma_red(self, v): self._update_model_state('pre_gamma_red', v)
    def on_cs_pre_gamma_green(self, v): self._update_model_state('pre_gamma_green', v)
    def on_cs_pre_gamma_blue(self, v): self._update_model_state('pre_gamma_blue', v)
    def on_cs_post_gamma_red(self, v): self._update_model_state('post_gamma_red', v)
    def on_cs_post_gamma_green(self, v): self._update_model_state('post_gamma_green', v)
    def on_cs_post_gamma_blue(self, v): self._update_model_state('post_gamma_blue', v)
    def on_cs_two_pass(self, v): self._update_model_state('two_pass', v)

    def on_cs_capture_flag(self, v):
        if v:
            self.known_embeddings = []
            self.auto_capture_mode = True
            print("[!] 已重置捕获模式：将锁定画面下一帧出现的首个人脸")
            self.get_control_sheet().capture_flag.set_flag(False)

    def _update_model_state(self, attr, val):
        state = self.get_state()
        if state.model_state is not None:
            setattr(state.model_state, attr, val)
            self.save_state()
            self.reemit_frame_signal.send()

    def on_cs_device(self, idx, device):
        """处理设备切换逻辑"""
        state, cs = self.get_state(), self.get_control_sheet()
        if state.device != device:
            state.device = device
            self.save_state()
            self.restart()
        else:
            # 2026 修复：如果设备相同但列表未刷新，则在此补全模型列表
            models_info = DFLive.get_available_models_info(self.dfm_models_path)
            cs.model.enable()
            cs.model.set_choices(models_info, none_choice_name='@misc.menu_select')
            cs.model.select(state.model)

    def on_cs_model(self, idx, model):
        """处理模型加载逻辑"""
        state, cs = self.get_state(), self.get_control_sheet()
        if model is not None:
            # 初始化该模型特定的状态
            if model.get_name() not in state.models_state:
                state.models_state[model.get_name()] = ModelState()
            
            state.model = model
            state.model_state = state.models_state[model.get_name()]
            
            # 立即激活 UI 配置
            self._update_ui_configs()
            
            # 启动模型推理引擎初始化
            self.dfm_model_initializer = DFLive.DFMModel_from_info(state.model, state.device)
            self.set_busy(True)
            
            # 异步加载身份锁定库
            if self.rec_model is None:
            #  self.load_target_identities()
        else:
            state.model = None
            self.save_state()
            self.restart()

    def _update_ui_configs(self):
        """同步后端状态到前端控件"""
        state, cs = self.get_state(), self.get_control_sheet()
        ms = state.model_state
        if ms is None: return

        cs.swap_all_faces.enable(); cs.swap_all_faces.set_flag(ms.swap_all_faces)
        cs.two_pass.enable(); cs.two_pass.set_flag(ms.two_pass)
        cs.capture_flag.enable()
        
        cs.face_id.enable(); cs.face_id.set_config(lib_csw.Number.Config(min=0, max=10, step=1))
        cs.face_id.set_number(ms.face_id)

        cs.morph_factor.enable(); cs.morph_factor.set_config(lib_csw.Number.Config(min=0, max=1, step=0.01, decimals=2))
        cs.morph_factor.set_number(ms.morph_factor)

        cs.presharpen_amount.enable(); cs.presharpen_amount.set_config(lib_csw.Number.Config(min=0, max=10, step=0.1))
        cs.presharpen_amount.set_number(ms.presharpen_amount)

        gamma_cfg = lib_csw.Number.Config(min=0.01, max=4.0, step=0.01, decimals=2)
        for attr in ['pre_gamma_red', 'pre_gamma_green', 'pre_gamma_blue', 'post_gamma_red', 'post_gamma_green', 'post_gamma_blue']:
            c = getattr(cs, attr)
            c.enable()
            c.set_config(gamma_cfg)
            c.set_number(getattr(ms, attr))

    # 此处应包含 load_target_identities 以及各个 on_cs_xxx 属性更新方法的具体实现...

    def on_tick(self):
        state, cs = self.get_state(), self.get_control_sheet()

        # 1. 模型初始化逻辑 (整合原版 UI 状态下发)
        if self.dfm_model_initializer is not None:
            events = self.dfm_model_initializer.process_events()

            if events.prev_status_downloading:
                self.set_busy(True)
                cs.model_dl_progress.disable()

            if events.new_status_downloading:
                self.set_busy(False)
                cs.model_dl_progress.enable()
                cs.model_dl_progress.set_config(lib_csw.Progress.Config(title='@FaceSwapDFM.downloading_model'))
                cs.model_dl_progress.set_progress(0)

            elif events.new_status_initialized:
                self.dfm_model = events.dfm_model
                self.dfm_model_initializer = None
                model_width, model_height = self.dfm_model.get_input_res()

                # 下发模型信息与 UI 配置 (解决按键不显示)
                cs.model_info_label.enable()
                cs.model_info_label.set_config(lib_csw.InfoLabel.Config(info_icon=True, info_lines=[
                    f'@FaceSwapDFM.model_information', '', f'@FaceSwapDFM.filename', 
                    f'{self.dfm_model.get_model_path().name}', '', f'@FaceSwapDFM.resolution', f'{model_width}x{model_height}']))

                # 激活所有控制滑块 (此处直接调用我们之前写的补全函数)
                self._update_ui_configs() 
                self.set_busy(False)
                self.reemit_frame_signal.send()

            elif events.new_status_error:
                self.set_busy(False)
                cs.model_dl_error.enable()
                cs.model_dl_error.set_error(events.error)

            if events.download_progress is not None:
                cs.model_dl_progress.set_progress(events.download_progress)
            return

        # 2. 核心图像处理循环
        if self.pending_bcd is None:
            bcd = self.bc_in.read(timeout=0.005)
            if bcd is not None:
                bcd.assign_weak_heap(self.weak_heap)
                ms = state.model_state
                dfm_model = self.dfm_model
                
                if all_is_not_None(dfm_model, ms):
                    # 确定是否需要自动捕获新人脸特征
                    should_capture = self.auto_capture_mode or (len(self.known_embeddings) == 0 and not ms.swap_all_faces)

                    for i, fsi in enumerate(bcd.get_face_swap_info_list()):
                        face_align_image = bcd.get_image(fsi.face_align_image_name)
                        if face_align_image is None: continue

                        # --- A. 身份识别与锁定逻辑 ---
                        is_match = False
                        if ms.swap_all_faces:
                            is_match = True
                        else:
                            if self.rec_model is None:
                                self.rec_model = InsightFaceRecognition(device='CPU')
                            
                            curr_emb = self.rec_model.extract(face_align_image)
                            if curr_emb is not None:
                                curr_emb = curr_emb.flatten()
                                if should_capture:
                                    self.known_embeddings = [curr_emb]
                                    self.auto_capture_mode = False
                                    print(f"[*] 2026自动锁定：已捕获画面中第一个人物 (Face ID: {i})")
                                    should_capture = False
                                    is_match = True
                                elif len(self.known_embeddings) > 0:
                                    # 对比特征
                                    dist = np.dot(self.known_embeddings[0], curr_emb)
                                    if dist > 0.65: is_match = True

                        # --- B. 换脸推理逻辑 (仅在匹配时执行) ---
                        if is_match:
                            # 预处理 (Sharpen & Pre-Gamma)
                            fai_ip = ImageProcessor(face_align_image)
                            if ms.presharpen_amount != 0:
                                fai_ip.gaussian_sharpen(sigma=1.0, power=ms.presharpen_amount)
                            fai_ip.gamma(ms.pre_gamma_red, ms.pre_gamma_green, ms.pre_gamma_blue)
                            face_align_image = fai_ip.get_image('HWC')

                            # 模型推理 (第一遍)
                            out = dfm_model.convert(face_align_image, morph_factor=ms.morph_factor)
                            celeb_face, celeb_face_mask, face_align_mask = out[0][0], out[1][0], out[2][0]

                            # Two Pass 逻辑
                            if ms.two_pass:
                                out = dfm_model.convert(celeb_face, morph_factor=ms.morph_factor)
                                celeb_face, celeb_face_mask = out[0][0], out[1][0]

                            # 后处理 (Post-Gamma)
                            if ms.post_gamma_red != 1.0 or ms.post_gamma_blue != 1.0 or ms.post_gamma_green != 1.0:
                                celeb_face = ImageProcessor(celeb_face).gamma(ms.post_gamma_red, ms.post_gamma_green, ms.post_gamma_blue).get_image('HWC')

                            # --- C. 核心修复：数据格式化与槽位写入 ---
                            def finalize(x, is_mask=False):
                                # 解决分辨率与类型报错 (uint8)
                                if x.shape[0] != 320: x = cv2.resize(x, (320, 320))
                                if x.dtype != np.uint8:
                                    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
                                if is_mask and x.ndim == 2: x = x[..., np.newaxis]
                                return x

                            # 规范槽位命名，确保 Merger 能识别
                            fsi.face_align_mask_name = f'{fsi.face_align_image_name}_mask'
                            fsi.face_swap_image_name = f'{fsi.face_align_image_name}_swapped'
                            fsi.face_swap_mask_name  = f'{fsi.face_swap_image_name}_mask'

                            # 写入 uint8 数据到 BCD
                            bcd.set_image(fsi.face_align_mask_name, finalize(face_align_mask, True))
                            bcd.set_image(fsi.face_swap_image_name, finalize(celeb_face))
                            bcd.set_image(fsi.face_swap_mask_name, finalize(celeb_face_mask, True))

                self.pending_bcd = bcd

        # 3. 输出推送
        if self.pending_bcd is not None:
            if self.bc_out.is_full_read(1):
                self.bc_out.write(self.pending_bcd)
                self.pending_bcd = None
            else:
                time.sleep(0.001)

class Sheet:
    class Host(lib_csw.Sheet.Host):
        def __init__(self):
            super().__init__()
            self.model = lib_csw.DynamicSingleSwitch.Client(); self.device = lib_csw.DynamicSingleSwitch.Client()
            self.model_info_label = lib_csw.InfoLabel.Client(); self.model_dl_progress = lib_csw.Progress.Client(); self.model_dl_error = lib_csw.Error.Client()
            self.capture_flag = lib_csw.Flag.Client(); self.swap_all_faces = lib_csw.Flag.Client(); self.two_pass = lib_csw.Flag.Client()
            self.face_id = lib_csw.Number.Client(); self.morph_factor = lib_csw.Number.Client(); self.presharpen_amount = lib_csw.Number.Client()
            self.pre_gamma_red = lib_csw.Number.Client(); self.pre_gamma_green = lib_csw.Number.Client(); self.pre_gamma_blue = lib_csw.Number.Client()
            self.post_gamma_red = lib_csw.Number.Client(); self.post_gamma_green = lib_csw.Number.Client(); self.post_gamma_blue = lib_csw.Number.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.model = lib_csw.DynamicSingleSwitch.Host(); self.device = lib_csw.DynamicSingleSwitch.Host()
            self.model_info_label = lib_csw.InfoLabel.Host(); self.model_dl_progress = lib_csw.Progress.Host(); self.model_dl_error = lib_csw.Error.Host()
            self.capture_flag = lib_csw.Flag.Host(); self.swap_all_faces = lib_csw.Flag.Host(); self.two_pass = lib_csw.Flag.Host()
            self.face_id = lib_csw.Number.Host(); self.morph_factor = lib_csw.Number.Host(); self.presharpen_amount = lib_csw.Number.Host()
            self.pre_gamma_red = lib_csw.Number.Host(); self.pre_gamma_green = lib_csw.Number.Host(); self.pre_gamma_blue = lib_csw.Number.Host()
            self.post_gamma_red = lib_csw.Number.Host(); self.post_gamma_green = lib_csw.Number.Host(); self.post_gamma_blue = lib_csw.Number.Host()

class ModelState:
    def __init__(self):
        self.swap_all_faces = False; self.face_id = 0; self.morph_factor = 0.75; self.presharpen_amount = 0.0
        self.pre_gamma_red = 1.0; self.pre_gamma_green = 1.0; self.pre_gamma_blue = 1.0
        self.post_gamma_red = 1.0; self.post_gamma_green = 1.0; self.post_gamma_blue = 1.0
        self.two_pass = False

class WorkerState(BackendWorkerState):
    def __init__(self):
        super().__init__()
        self.device = None; self.model = None; self.models_state = {}; self.model_state = None
