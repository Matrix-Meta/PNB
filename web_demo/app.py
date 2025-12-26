import os
import io
import sys
import traceback
import base64
import logging
import time
import psutil
import argparse
import struct
import numpy as np

# Configure Logging
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pnb_web.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PNB-Web")

try:
    import dpnp
    import dpctl
    if dpctl.has_gpu_devices():
        xp = dpnp
        logger.info("Backend: dpnp (Intel GPU Accelerated)")
    else:
        xp = np
        logger.info("Backend: numpy (CPU - No GPU found)")
except ImportError:
    xp = np
    logger.info("Backend: numpy (CPU - dpnp not installed)")

from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

# --- Native PNB Inference Engine (NumPy/DPNP) ---

class NativePNB:
    def __init__(self, path):
        self.xp = xp # Bind the active backend
        self.load_ckpt(path)
        logger.info(f"Native Model initialized: {self.hidden_dim} hidden units on {self.xp.__name__}")

    def _read_vec_f32(self, f):
        n = struct.unpack("<Q", f.read(8))[0]
        # Read to CPU numpy first, then transfer if needed
        arr = np.frombuffer(f.read(int(n) * 4), dtype="<f4").copy()
        return self.xp.array(arr) if self.xp == dpnp else arr

    def load_ckpt(self, path):
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != b"PNBCKPT1": raise ValueError("Invalid magic")
            f.seek(12, 0) # Skip version
            self.batch_size = struct.unpack("<Q", f.read(8))[0]
            self.seq_len = struct.unpack("<Q", f.read(8))[0]
            self.input_dim = struct.unpack("<Q", f.read(8))[0]
            self.hidden_dim = struct.unpack("<Q", f.read(8))[0]
            self.output_dim = struct.unpack("<Q", f.read(8))[0]
            self.md_group_size = struct.unpack("<Q", f.read(8))[0]
            self.md_group_count = struct.unpack("<Q", f.read(8))[0]
            
            # Skip training stats (40 bytes)
            f.seek(40, 1)
            
            # Glial State (12 floats)
            glial_data = struct.unpack("<12f", f.read(48))
            self.current_threshold = glial_data[0] # current_threshold
            
            # Target Sparsity & Snapshots
            self.target_sparsity = struct.unpack("<f", f.read(4))[0]
            self.glial_snapshot = struct.unpack("<f", f.read(4))[0]
            f.seek(4, 1) # Skip hipp_learned_th
            
            # Neuromodulator & MD Config
            f.seek(11 * 4, 1) # Skip nm_w, nm_last
            
            # Convert scalars to backend arrays for compatibility
            self.md_vigilance = self.xp.array(struct.unpack("<f", f.read(4))[0])
            self.md_tau = self.xp.array(struct.unpack("<f", f.read(4))[0])
            self.md_alpha = self.xp.array(struct.unpack("<f", f.read(4))[0])
            self.md_novelty_p = self.xp.array(struct.unpack("<f", f.read(4))[0])
            self.md_novelty_s = self.xp.array(struct.unpack("<f", f.read(4))[0])
            
            f.seek(1, 1) # Skip bias flags
            
            # Group Offsets
            self.md_group_offsets = self._read_vec_f32(f)
            
            # Weights
            def read_mat(r, c): return self._read_vec_f32(f).reshape(r, c)
            
            # Helper to pre-quantize weights
            def quantize_weight(w):
                beta = self.xp.mean(self.xp.abs(w)) + 1e-6
                wq = self.xp.clip(self.xp.round(w / beta), -1, 1).astype(self.xp.int8)
                return wq, beta

            # Read FP32 weights, quantize to INT8 immediately, discard FP32
            w_in_f32 = read_mat(self.input_dim, self.hidden_dim)
            self.w_in, self.beta_in = quantize_weight(w_in_f32)
            self.b_in = self._read_vec_f32(f)
            
            w_m1_f32 = read_mat(self.hidden_dim, self.hidden_dim)
            self.w_m1, self.beta_m1 = quantize_weight(w_m1_f32)
            self.b_m1 = self._read_vec_f32(f)
            
            w_m2_f32 = read_mat(self.hidden_dim, self.hidden_dim)
            self.w_m2, self.beta_m2 = quantize_weight(w_m2_f32)
            self.b_m2 = self._read_vec_f32(f)
            
            w_out_f32 = read_mat(self.hidden_dim, self.output_dim)
            self.w_out, self.beta_out = quantize_weight(w_out_f32)
            self.b_out = self._read_vec_f32(f)
            
            self.w_fast = read_mat(self.input_dim, self.hidden_dim)
            self.w_slow = read_mat(self.input_dim, self.hidden_dim)
            
            # Calculated static memory size (Weights are now 1 byte instead of 4)
            self.mem_mb = (self.w_in.size + self.w_m1.size + self.w_m2.size + self.w_out.size) * 1 / 1024 / 1024

    def bitlinear(self, x, w_int8, beta, b):
        # Weight Quantization: Done at load time! w_int8 is ready.
        # 8-bit Input Quantization
        gamma = self.xp.max(self.xp.abs(x)) / 127.0 + 1e-6
        xq = self.xp.clip(self.xp.round(x / gamma), -127, 127)
        # Compute: int8 @ int8 (or float depending on backend)
        y = (xq @ w_int8) * (beta * gamma)
        return y + b

    def layer_norm(self, x):
        mean = self.xp.mean(x, axis=1, keepdims=True)
        var = self.xp.var(x, axis=1, keepdims=True)
        return (x - mean) / self.xp.sqrt(var + 1e-5)

    def mdsilu(self, x, novelty):
        # Apply group offsets to threshold
        # Need to ensure idx is compatible with backend
        idx = self.xp.minimum(self.xp.arange(self.hidden_dim) // self.md_group_size, self.md_group_count - 1).astype(self.xp.int64)
        offsets = self.md_group_offsets[idx]
        
        # Novelty shift (Neuromodulation)
        # Simplified novelty shaping (matching ONNX logic)
        n_shaped = self.xp.power(novelty, self.md_novelty_p)
        if self.md_novelty_s > 0:
            n_shaped = (1.0 - self.xp.exp(-self.md_novelty_s * n_shaped)) / (1.0 - self.xp.exp(-self.md_novelty_s))
        
        eff_th = self.glial_snapshot + offsets + self.md_vigilance * self.xp.clip(n_shaped, 0, 1)
        
        # Gating
        gate = 1.0 / (1.0 + self.xp.exp(-(x - eff_th) / self.md_tau))
        # SiLU
        silu = x * (1.0 / (1.0 + self.xp.exp(-x)))
        # Leaky
        leaky = self.md_alpha * self.xp.minimum(0, x - eff_th)
        
        return silu * gate + leaky

    def forward(self, x):
        # Ensure input is on correct device
        if self.xp == dpnp and not isinstance(x, dpnp.ndarray):
            x = self.xp.array(x)

        # Familiarity calculation (Neuromodulator)
        w_mem = self.w_fast + self.w_slow
        w_sum = self.xp.sum(w_mem, axis=1)
        dot = self.xp.sum(x * w_sum)
        norm_x = self.xp.sqrt(self.xp.sum(x*x) + 1e-8)
        norm_w = self.xp.sqrt(self.xp.sum(w_mem*w_mem) + 1e-8)
        novelty = 1.0 - self.xp.clip(dot / (norm_x * norm_w + 1e-8), 0, 1)

        # Layers
        h = self.bitlinear(x, self.w_in, self.beta_in, self.b_in)
        h = self.mdsilu(self.layer_norm(h), novelty)
        
        h = self.bitlinear(h, self.w_m1, self.beta_m1, self.b_m1)
        h = self.mdsilu(self.layer_norm(h), novelty)
        
        h = self.bitlinear(h, self.w_m2, self.beta_m2, self.b_m2)
        h = self.mdsilu(self.layer_norm(h), novelty)
        
        logits = self.bitlinear(h, self.w_out, self.beta_out, self.b_out)
        
        # Convert back to numpy for consistent return interface if needed
        if self.xp == dpnp:
            return logits[0].asnumpy(), float(novelty)
        return logits[0], float(novelty)

# --- ONNX Inference Wrapper ---

class OnnxPNB:
    def __init__(self, path):
        import onnxruntime as ort
        import onnx
        self.session = ort.InferenceSession(path)
        # Calculate size
        model = onnx.load(path)
        total_bytes = sum(np.prod(i.dims) * 4 for i in model.graph.initializer)
        self.mem_mb = total_bytes / 1024 / 1024

    def forward(self, x):
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: x})
        return outputs[0][0], 0.0 # ONNX doesn't return novelty by default here

# --- Flask Server ---

model_engine = None

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_engine is None: return jsonify({'error': 'No model'}), 500
    try:
        t_start = time.perf_counter()
        data = request.json
        image_data = data['image'].split(",")[1]
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Preprocess
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (0, 0, 0))
            bg.paste(img, mask=img.split()[3])
            img = bg
        img = img.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
        
        # Auto-centering (Center of Mass)
        img_arr = np.array(img)
        try:
            from scipy.ndimage import center_of_mass, shift
            cy, cx = center_of_mass(img_arr)
            # If image is empty (all black), center_of_mass returns NaNs or throws error
            if not np.isnan(cy) and not np.isnan(cx):
                dy = 14 - cy
                dx = 14 - cx
                img_arr = shift(img_arr, [dy, dx], cval=0.0)
                # Clip to valid range after shift interpolation
                img_arr = np.clip(img_arr, 0, 255)
        except ImportError:
            # Fallback if scipy is not installed (simple bounding box centering could be done here, 
            # but for now we skip to keep dependencies minimal)
            pass
        except Exception as e:
            # Handle cases like empty images
            pass

        x = img_arr.astype(np.float32).reshape(1, 784) / 255.0
        
        # Inference
        t_infer_start = time.perf_counter()
        logits, novelty = model_engine.forward(x)
        t_infer_end = time.perf_counter()
        
        probs = softmax(logits)
        prediction = int(np.argmax(probs))
        
        return jsonify({
            'prediction': prediction,
            'probabilities': probs.tolist(),
            'confidence': float(probs[prediction]),
            'stats': {
                'inference_ms': (t_infer_end - t_infer_start) * 1000,
                'model_mem_mb': model_engine.mem_mb,
                'novelty': float(novelty)
            }
        })
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../exports/mnist_surrogate_best_v11.onnx")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.model.endswith('.bin') or args.model.endswith('.ckpt'):
        logger.info("Using Native PNB Engine (NumPy)")
        model_engine = NativePNB(args.model)
    else:
        logger.info("Using ONNX Runtime Engine")
        model_engine = OnnxPNB(args.model)

    app.run(host='0.0.0.0', port=args.port, debug=False)