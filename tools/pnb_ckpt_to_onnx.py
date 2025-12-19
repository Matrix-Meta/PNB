#!/usr/bin/env python3
import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Ckpt:
    batch_size: int
    seq_len: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    md_group_size: int
    md_group_count: int

    glial_threshold_snapshot: float
    md_vigilance: float
    md_tau: float
    md_alpha: float
    md_novelty_power: float
    md_novelty_sharpness: float
    md_group_offsets: np.ndarray

    proj_in_w: np.ndarray
    proj_in_b: np.ndarray
    proj_mid1_w: np.ndarray
    proj_mid1_b: np.ndarray
    proj_mid2_w: np.ndarray
    proj_mid2_b: np.ndarray
    proj_out_w: np.ndarray
    proj_out_b: np.ndarray

    w_fast: np.ndarray
    w_slow: np.ndarray


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError("unexpected EOF")
    return b


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_f32(f) -> float:
    return struct.unpack("<f", _read_exact(f, 4))[0]


def _read_vec_f32(f) -> np.ndarray:
    n = _read_u64(f)
    raw = _read_exact(f, int(n) * 4)
    return np.frombuffer(raw, dtype="<f4").copy()


def load_ckpt(path: Path) -> Ckpt:
    with path.open("rb") as f:
        magic = _read_exact(f, 8)
        if magic != b"PNBCKPT1":
            raise ValueError("bad magic (expected PNBCKPT1)")
        ver = _read_u32(f)
        if ver != 1:
            raise ValueError(f"unsupported version: {ver}")

        batch_size = _read_u64(f)
        seq_len = _read_u64(f)
        input_dim = _read_u64(f)
        hidden_dim = _read_u64(f)
        output_dim = _read_u64(f)
        md_group_size = _read_u64(f)
        md_group_count = _read_u64(f)

        _epoch = _read_u32(f)
        _seed = _read_u32(f)
        _lr = _read_f32(f)
        _train_acc = _read_f32(f)
        _train_loss = _read_f32(f)
        _train_unc = _read_f32(f)
        _val_acc = _read_f32(f)
        _val_loss = _read_f32(f)
        _val_unc = _read_f32(f)
        _flags = _read_u32(f)

        _glial_current_threshold = _read_f32(f)
        _glial_current_lr = _read_f32(f)
        _glial_prev_error = _read_f32(f)
        _glial_error_integral = _read_f32(f)
        _glial_threshold_velocity = _read_f32(f)
        _glial_sparsity_ema = _read_f32(f)
        _glial_error_ema = _read_f32(f)
        _glial_last_change_ratio = _read_f32(f)
        _glial_stable_count = struct.unpack("<i", _read_exact(f, 4))[0]
        _glial_is_first_call = struct.unpack("<i", _read_exact(f, 4))[0]
        _glial_current_noise_gain = _read_f32(f)
        _glial_last_sparsity = _read_f32(f)

        _target_sparsity = _read_f32(f)
        glial_threshold_snapshot = _read_f32(f)
        _hipp_learned_th = _read_f32(f)

        _nm_w = [_read_f32(f) for _ in range(4)]
        _nm_last = [_read_f32(f) for _ in range(7)]

        md_vigilance = _read_f32(f)
        md_tau = _read_f32(f)
        md_alpha = _read_f32(f)
        md_novelty_power = _read_f32(f)
        md_novelty_sharpness = _read_f32(f)

        _use_bias_flags = struct.unpack("<B", _read_exact(f, 1))[0]

        md_group_offsets = _read_vec_f32(f)
        proj_in_weight = _read_vec_f32(f)
        proj_in_bias = _read_vec_f32(f)
        proj_mid1_weight = _read_vec_f32(f)
        proj_mid1_bias = _read_vec_f32(f)
        proj_mid2_weight = _read_vec_f32(f)
        proj_mid2_bias = _read_vec_f32(f)
        proj_out_weight = _read_vec_f32(f)
        proj_out_bias = _read_vec_f32(f)
        w_fast = _read_vec_f32(f)
        w_slow = _read_vec_f32(f)
        _ssm_a = _read_vec_f32(f)
        _ssm_state = _read_vec_f32(f)

    In = int(input_dim)
    H = int(hidden_dim)
    O = int(output_dim)

    def mat(v: np.ndarray, rows: int, cols: int) -> np.ndarray:
        if v.size != rows * cols:
            raise ValueError(f"bad matrix size: expected {rows*cols}, got {v.size}")
        return v.reshape((rows, cols))

    return Ckpt(
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        input_dim=In,
        hidden_dim=H,
        output_dim=O,
        md_group_size=int(md_group_size),
        md_group_count=int(md_group_count),
        glial_threshold_snapshot=float(glial_threshold_snapshot),
        md_vigilance=float(md_vigilance),
        md_tau=float(md_tau),
        md_alpha=float(md_alpha),
        md_novelty_power=float(md_novelty_power),
        md_novelty_sharpness=float(md_novelty_sharpness),
        md_group_offsets=md_group_offsets.astype(np.float32),
        proj_in_w=mat(proj_in_weight.astype(np.float32), In, H),
        proj_in_b=proj_in_bias.astype(np.float32),
        proj_mid1_w=mat(proj_mid1_weight.astype(np.float32), H, H),
        proj_mid1_b=proj_mid1_bias.astype(np.float32),
        proj_mid2_w=mat(proj_mid2_weight.astype(np.float32), H, H),
        proj_mid2_b=proj_mid2_bias.astype(np.float32),
        proj_out_w=mat(proj_out_weight.astype(np.float32), H, O),
        proj_out_b=proj_out_bias.astype(np.float32),
        w_fast=mat(w_fast.astype(np.float32), In, H),
        w_slow=mat(w_slow.astype(np.float32), In, H),
    )


def build_onnx(ckpt: Ckpt, out_path: Path, opset: int = 18) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    In = ckpt.input_dim
    H = ckpt.hidden_dim
    O = ckpt.output_dim
    gs = max(1, ckpt.md_group_size)
    gc = max(1, ckpt.md_group_count)

    def const(name: str, value: np.ndarray | float | int, dtype=TensorProto.FLOAT):
        if isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.array(value)
        if dtype == TensorProto.FLOAT:
            arr = arr.astype(np.float32)
        elif dtype == TensorProto.INT64:
            arr = arr.astype(np.int64)
        else:
            raise ValueError("unsupported dtype")
        return numpy_helper.from_array(arr, name=name)

    nodes = []
    inits = []
    value_infos = []

    X = "input"
    logits = "logits"

    inputs = [
        helper.make_tensor_value_info(X, TensorProto.FLOAT, ["batch", In]),
    ]
    outputs = [
        helper.make_tensor_value_info(logits, TensorProto.FLOAT, ["batch", O]),
    ]

    eps6 = "c_eps6"
    eps8 = "c_eps8"
    eps_ln = "c_epsln"
    c127 = "c127"
    cneg127 = "cneg127"
    c1 = "c1"
    c0 = "c0"
    cneg1 = "cneg1"

    inits.extend(
        [
            const(eps6, 1e-6),
            const(eps8, 1e-8),
            const(eps_ln, 1e-5),
            const(c127, 127.0),
            const(cneg127, -127.0),
            const(c1, 1.0),
            const(c0, 0.0),
            const(cneg1, -1.0),
        ]
    )

    def reduce_op(name: str, op_type: str, x: str, axes: list[int], keepdims: int):
        y = name
        axes_i64 = [int(a) for a in axes]
        if opset >= 13:
            axes_name = f"{name}_axes"
            inits.append(
                const(axes_name, np.array(axes_i64, dtype=np.int64), dtype=TensorProto.INT64)
            )
            nodes.append(helper.make_node(op_type, [x, axes_name], [y], keepdims=int(keepdims)))
        else:
            nodes.append(
                helper.make_node(op_type, [x], [y], axes=axes_i64, keepdims=int(keepdims))
            )
        return y

    def reduce_mean(name, x, axes):
        return reduce_op(name, "ReduceMean", x, axes=axes, keepdims=1)

    def reduce_sum(name, x, axes, keepdims=1):
        return reduce_op(name, "ReduceSum", x, axes=axes, keepdims=keepdims)

    def reduce_max(name, x, axes):
        return reduce_op(name, "ReduceMax", x, axes=axes, keepdims=1)

    def unsqueeze(name: str, x: str, axes: list[int]):
        y = name
        axes_i64 = [int(a) for a in axes]
        if opset >= 13:
            axes_name = f"{name}_axes"
            inits.append(
                const(axes_name, np.array(axes_i64, dtype=np.int64), dtype=TensorProto.INT64)
            )
            nodes.append(helper.make_node("Unsqueeze", [x, axes_name], [y]))
        else:
            nodes.append(helper.make_node("Unsqueeze", [x], [y], axes=axes_i64))
        return y

    def layer_norm(prefix, x):
        mean = reduce_mean(prefix + "_mean", x, axes=[1])
        centered = prefix + "_centered"
        nodes.append(helper.make_node("Sub", [x, mean], [centered]))
        sq = prefix + "_sq"
        nodes.append(helper.make_node("Mul", [centered, centered], [sq]))
        var = reduce_mean(prefix + "_var", sq, axes=[1])
        var_eps = prefix + "_var_eps"
        nodes.append(helper.make_node("Add", [var, eps_ln], [var_eps]))
        std = prefix + "_std"
        nodes.append(helper.make_node("Sqrt", [var_eps], [std]))
        rstd = prefix + "_rstd"
        nodes.append(helper.make_node("Div", [c1, std], [rstd]))
        y = prefix + "_ln"
        nodes.append(helper.make_node("Mul", [centered, rstd], [y]))
        return y

    def bitlinear(prefix, x, W_name, B_name, out_dim):
        absw = prefix + "_absw"
        nodes.append(helper.make_node("Abs", [W_name], [absw]))
        beta = reduce_mean(prefix + "_beta0", absw, axes=[0, 1])
        beta_eps = prefix + "_beta"
        nodes.append(helper.make_node("Add", [beta, eps6], [beta_eps]))

        wscaled = prefix + "_wscaled"
        nodes.append(helper.make_node("Div", [W_name, beta_eps], [wscaled]))
        wround = prefix + "_wround"
        nodes.append(helper.make_node("Round", [wscaled], [wround]))
        wq = prefix + "_wq"
        nodes.append(helper.make_node("Clip", [wround, cneg1, c1], [wq]))

        absx = prefix + "_absx"
        nodes.append(helper.make_node("Abs", [x], [absx]))
        maxx = reduce_max(prefix + "_maxx0", absx, axes=[1])
        maxx_div = prefix + "_maxx_div"
        nodes.append(helper.make_node("Div", [maxx, c127], [maxx_div]))
        gamma = prefix + "_gamma"
        nodes.append(helper.make_node("Add", [maxx_div, eps6], [gamma]))

        xscaled = prefix + "_xscaled"
        nodes.append(helper.make_node("Div", [x, gamma], [xscaled]))
        xround = prefix + "_xround"
        nodes.append(helper.make_node("Round", [xscaled], [xround]))
        xq = prefix + "_xq"
        nodes.append(helper.make_node("Clip", [xround, cneg127, c127], [xq]))

        yint = prefix + "_yint"
        nodes.append(helper.make_node("MatMul", [xq, wq], [yint]))
        yscaled = prefix + "_yscaled0"
        nodes.append(helper.make_node("Mul", [yint, beta_eps], [yscaled]))
        yscaled2 = prefix + "_yscaled"
        nodes.append(helper.make_node("Mul", [yscaled, gamma], [yscaled2]))

        y = prefix + "_y"
        nodes.append(helper.make_node("Add", [yscaled2, B_name], [y]))
        value_infos.append(
            helper.make_tensor_value_info(y, TensorProto.FLOAT, ["batch", out_dim])
        )
        return y

    def mdsilu(prefix, x, base_th_name, fam_b, Hdim):
        group_offsets = prefix + "_group_offsets"
        group_idx = prefix + "_group_idx"
        inits.append(const(group_offsets, ckpt.md_group_offsets))
        idx = np.minimum(np.arange(Hdim) // gs, gc - 1).astype(np.int64)
        inits.append(const(group_idx, idx, dtype=TensorProto.INT64))

        go = prefix + "_go"
        nodes.append(helper.make_node("Gather", [group_offsets, group_idx], [go], axis=0))
        go2 = unsqueeze(prefix + "_go2", go, axes=[0])
        xshape = prefix + "_xshape"
        nodes.append(helper.make_node("Shape", [x], [xshape]))
        go_b = prefix + "_go_b"
        nodes.append(helper.make_node("Expand", [go2, xshape], [go_b]))

        novelty = prefix + "_novelty"
        nodes.append(helper.make_node("Sub", [c1, fam_b], [novelty]))
        p_name = prefix + "_p"
        sharp_name = prefix + "_sharp"
        vig_name = prefix + "_vig"
        tau_name = prefix + "_tau"
        alpha_name = prefix + "_alpha"
        inits.append(const(p_name, ckpt.md_novelty_power))
        inits.append(const(sharp_name, ckpt.md_novelty_sharpness))
        inits.append(const(vig_name, ckpt.md_vigilance))
        inits.append(const(tau_name, ckpt.md_tau))
        inits.append(const(alpha_name, ckpt.md_alpha))

        novelty_pow = prefix + "_novelty_pow"
        nodes.append(helper.make_node("Pow", [novelty, p_name], [novelty_pow]))

        novelty_shaped = novelty_pow
        if ckpt.md_novelty_sharpness > 0.0:
            exp_arg = prefix + "_exp_arg"
            neg_sharp = prefix + "_neg_sharp"
            inits.append(const(neg_sharp, -ckpt.md_novelty_sharpness))
            nodes.append(helper.make_node("Mul", [neg_sharp, novelty_pow], [exp_arg]))
            expv = prefix + "_expv"
            nodes.append(helper.make_node("Exp", [exp_arg], [expv]))
            num = prefix + "_num"
            nodes.append(helper.make_node("Sub", [c1, expv], [num]))

            denom_const = float(1.0 - np.exp(-ckpt.md_novelty_sharpness))
            denom_name = prefix + "_denom"
            inits.append(const(denom_name, denom_const))
            novelty_shaped = prefix + "_novelty_shaped"
            nodes.append(helper.make_node("Div", [num, denom_name], [novelty_shaped]))

        novelty_clip = prefix + "_novelty_clip"
        nodes.append(helper.make_node("Clip", [novelty_shaped, c0, c1], [novelty_clip]))
        novelty_shift = prefix + "_novelty_shift"
        nodes.append(helper.make_node("Mul", [vig_name, novelty_clip], [novelty_shift]))

        th0 = prefix + "_th0"
        nodes.append(helper.make_node("Add", [base_th_name, go_b], [th0]))
        eff_th = prefix + "_eff_th"
        nodes.append(helper.make_node("Add", [th0, novelty_shift], [eff_th]))

        x_shifted = prefix + "_x_shifted"
        nodes.append(helper.make_node("Sub", [x, eff_th], [x_shifted]))

        x_div = prefix + "_x_div"
        nodes.append(helper.make_node("Div", [x_shifted, tau_name], [x_div]))
        gate = prefix + "_gate"
        nodes.append(helper.make_node("Sigmoid", [x_div], [gate]))

        sigx = prefix + "_sigx"
        nodes.append(helper.make_node("Sigmoid", [x], [sigx]))
        silu = prefix + "_silu"
        nodes.append(helper.make_node("Mul", [x, sigx], [silu]))

        min0 = prefix + "_min0"
        nodes.append(helper.make_node("Min", [c0, x_shifted], [min0]))
        leaky = prefix + "_leaky"
        nodes.append(helper.make_node("Mul", [alpha_name, min0], [leaky]))

        mul = prefix + "_mul"
        nodes.append(helper.make_node("Mul", [silu, gate], [mul]))
        y = prefix + "_y"
        nodes.append(helper.make_node("Add", [mul, leaky], [y]))
        return y

    # Constants / weights
    W_in = "W_in"
    b_in = "b_in"
    W_m1 = "W_m1"
    b_m1 = "b_m1"
    W_m2 = "W_m2"
    b_m2 = "b_m2"
    W_out = "W_out"
    b_out = "b_out"
    inits.extend(
        [
            const(W_in, ckpt.proj_in_w),
            const(b_in, ckpt.proj_in_b),
            const(W_m1, ckpt.proj_mid1_w),
            const(b_m1, ckpt.proj_mid1_b),
            const(W_m2, ckpt.proj_mid2_w),
            const(b_m2, ckpt.proj_mid2_b),
            const(W_out, ckpt.proj_out_w),
            const(b_out, ckpt.proj_out_b),
        ]
    )

    base_th = "base_th"
    inits.append(const(base_th, ckpt.glial_threshold_snapshot))

    # Familiarity uses batch[0] sample, matching current C++ behavior.
    idx0 = "idx0"
    inits.append(const(idx0, np.array([0], dtype=np.int64), dtype=TensorProto.INT64))
    X0 = "X0"
    nodes.append(helper.make_node("Gather", [X, idx0], [X0], axis=0))

    W_mem = (ckpt.w_fast + ckpt.w_slow).astype(np.float32)
    W_mem_name = "W_mem"
    inits.append(const(W_mem_name, W_mem))

    row_sum = reduce_sum("row_sum", W_mem_name, axes=[1], keepdims=0)
    x0_mul = "x0_mul"
    nodes.append(helper.make_node("Mul", [X0, row_sum], [x0_mul]))
    dot = reduce_sum("dot", x0_mul, axes=[1], keepdims=1)

    x0_sq = "x0_sq"
    nodes.append(helper.make_node("Mul", [X0, X0], [x0_sq]))
    xnorm_sq = reduce_sum("xnorm_sq", x0_sq, axes=[1], keepdims=1)
    xnorm_sq_eps = "xnorm_sq_eps"
    nodes.append(helper.make_node("Add", [xnorm_sq, eps8], [xnorm_sq_eps]))
    xnorm = "xnorm"
    nodes.append(helper.make_node("Sqrt", [xnorm_sq_eps], [xnorm]))

    mem_norm_const = float(np.sqrt(np.sum(W_mem * W_mem) + 1e-8))
    mem_norm = "mem_norm"
    inits.append(const(mem_norm, mem_norm_const))

    denom0 = "denom0"
    nodes.append(helper.make_node("Mul", [xnorm, mem_norm], [denom0]))
    denom = "denom"
    nodes.append(helper.make_node("Add", [denom0, eps8], [denom]))
    fam = "fam"
    nodes.append(helper.make_node("Div", [dot, denom], [fam]))
    fam_clip = "fam_clip"
    nodes.append(helper.make_node("Clip", [fam, c0, c1], [fam_clip]))

    # Broadcast fam_clip [1,1] -> [batch,1]
    xshape = "shapeX"
    nodes.append(helper.make_node("Shape", [X], [xshape]))
    start0 = "slice_start0"
    end1 = "slice_end1"
    axes0 = "slice_axes0"
    steps1 = "slice_steps1"
    inits.extend(
        [
            const(start0, np.array([0], dtype=np.int64), dtype=TensorProto.INT64),
            const(end1, np.array([1], dtype=np.int64), dtype=TensorProto.INT64),
            const(axes0, np.array([0], dtype=np.int64), dtype=TensorProto.INT64),
            const(steps1, np.array([1], dtype=np.int64), dtype=TensorProto.INT64),
        ]
    )
    batch_dim = "batch_dim"
    nodes.append(helper.make_node("Slice", [xshape, start0, end1, axes0, steps1], [batch_dim]))
    one_dim = "one_dim"
    inits.append(const(one_dim, np.array([1], dtype=np.int64), dtype=TensorProto.INT64))
    fam_shape = "fam_shape"
    nodes.append(helper.make_node("Concat", [batch_dim, one_dim], [fam_shape], axis=0))
    fam_b = "fam_b"
    nodes.append(helper.make_node("Expand", [fam_clip, fam_shape], [fam_b]))

    # Network
    h0 = bitlinear("in", X, W_in, b_in, H)
    h0_ln = layer_norm("ln0", h0)
    h0_act = mdsilu("act0", h0_ln, base_th, fam_b, H)

    h1 = bitlinear("m1", h0_act, W_m1, b_m1, H)
    h1_ln = layer_norm("ln1", h1)
    h1_act = mdsilu("act1", h1_ln, base_th, fam_b, H)

    h2 = bitlinear("m2", h1_act, W_m2, b_m2, H)
    h2_ln = layer_norm("ln2", h2)
    h2_act = mdsilu("act2", h2_ln, base_th, fam_b, H)

    out_y = bitlinear("out", h2_act, W_out, b_out, O)
    nodes.append(helper.make_node("Identity", [out_y], [logits]))

    graph = helper.make_graph(
        nodes=nodes,
        name="PNB_MNIST_Surrogate",
        inputs=inputs,
        outputs=outputs,
        initializer=inits,
        value_info=value_infos,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="PNB",
    )
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input .ckpt/.bin (PNBCKPT1)")
    ap.add_argument("--out", dest="out", required=True, help="output .onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    ckpt = load_ckpt(inp)
    build_onnx(ckpt, out, opset=args.opset)


if __name__ == "__main__":
    main()
