# PNB: Project Neuro-Bit (PNB-X)
### 基於極低位元量化與仿生動力學的高效能類腦運算框架
#### Quantized Brain-Inspired Computing with BitNet b1.58, Selective State-Space Models, and Spiking Dynamics

[![授權協議](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![學術基準](https://img.shields.io/badge/Architecture-PNB--X-red.svg)](#系統架構細節)
[![編譯器](https://img.shields.io/badge/Compiler-icpx_2025.3-0071C5.svg)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
[![硬體優化](https://img.shields.io/badge/Hardware-Intel_Arc_/_Xeon-0071C5.svg)](#硬體層級優化)

---

## 📝 1. 摘要 (Abstract)

在人工智慧領域，傳統的 Transformer 架構正面臨著注意力機制帶來的平方級複雜度 ($O(L^2)$) 以及動態隨機存取記憶體 (DRAM) 頻寬瓶頸的嚴峻挑戰。**Project Neuro-Bit (PNB)** 旨在探索一種結合神經科學原理與極致硬體優化的新型運算範式。

我們提出的 **PNB-X** 推論引擎，整合了 **BitNet b1.58 (三元量化)**、**選擇性狀態空間模型 (Selective SSM)** 以及 **脈衝神經網路 (SNN)**。透過模擬哺乳動物大腦的 **互補學習系統 (CLS)** 與 **膠質細胞穩態機制 (Glial Homeostasis)**，PNB-X 成功在維持 **1.58-bit** 極低權重精度的同時，於 **MNIST** 任務中達成了 **99.15%** 的辨識準確率。其實作核心基於 **Intel oneAPI/SYCL**，專為現代異構運算設備（Intel Arc/Xeon）設計，並針對 **XMX** 加速器進行了深度核心融合優化。

---

## 🔬 2. 理論背景與數學形式化 (Theoretical Foundations)

### 2.1 三元量化權重 (BitNet b1.58)
不同於傳統的二進位量化，PNB-X 採用三元權重矩陣 $W \in \{-1, 0, 1\}^d$，其量化過程遵循以下縮放轉換：
$$W_q = \text{sign}(W - \mathbb{E}[W])$$ 
$$\gamma = \frac{1}{nm} \sum_{i,j} |W_{i,j}|$$ 
$$\hat{W} = W_q \times \gamma$$ 
這種設計允許在推論時將矩陣乘法中的浮點乘法操作完全替換為**整數累加**，從而將計算密度提升一個量級。

### 2.2 選擇性狀態空間掃描 (Selective SSM)
為了捕捉長程時序依賴並保持線性時間複雜度，PNB-X 採用離散化的狀態空間方程：
$$h_k = \mathbf{\bar{A}}h_{k-1} + \mathbf{\bar{B}}x_k$$ 
$$y_k = \mathbf{C}h_k + \mathbf{D}x_k$$ 
其中，離散化矩陣 $\mathbf{\bar{A}}$ 與 $\mathbf{\bar{B}}$ 透過零階保持器 (ZOH) 定義：
$$\mathbf{\bar{A}} = \exp(\Delta \mathbf{A}), \quad \mathbf{\bar{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$ 
PNB-X 的創新之處在於 $\Delta, \mathbf{B}, \mathbf{C}$ 均為輸入相關的張量，這賦予了模型強大的內容感知過濾能力。

### 2.3 脈衝動力學與代理梯度 (Spiking Dynamics)
神經元模型採用簡化的 Leaky Integrate-and-Fire (LIF) 動力學：
$$v[t] = \alpha v[t-1] + \sum w_i S_i[t]$$ 
當 $v[t] > \theta$ 時發射脈衝。為了在反向傳播中處理不可微的階梯函數，我們引入了矩形核代理梯度：
$$\frac{\partial S}{\partial v} = \frac{1}{a} \text{rect}(\frac{v - \theta}{a})$$

---

## 🏛️ 3. 系統架構細節 (Architecture Deep Dive)

### 3.1 仿生新皮層 (The Neocortex Tier)
作為系統的長期知識庫，新皮層層級由多層 **BitLinear** 與 **Mamba-Scan** 組件構成。
*   **LayerNorm-First**: 在量化層前強制執行 LayerNorm，以消除激活值離散化後的數值不穩定性。
*   **SSM-Scan**: 實現了與序列長度 $L$ 成正比的推論成本，確保在超長序列感測任務中記憶體佔用保持恆定。

### 3.2 海馬體 CLS 記憶環路 (Hippocampal Loop)
整合了互補學習系統 (CLS) 理論，海馬體模組具備以下雙重職能：
*   **一發學習 (One-shot Encoding)**：利用 Hebbian 學習規則，迅速在快速權重 $W_{\text{fast}}$ 中建立特徵關聯。
*   **記憶固化 (Consolidation)**：模擬睡眠週期的權重同步過程：
$$W_{\text{slow}} \leftarrow W_{\text{slow}} + \eta_{\text{cls}} W_{\text{fast}}$$
$$W_{\text{fast}} \leftarrow W_{\text{fast}} \times (1 - \text{decay})$$

### 3.3 膠質細胞穩態控制器 (Glial Regulation)
膠質細胞負責監測全網活躍率 $\rho$，並透過內置的 PID 反饋環路動態調整激發閾值 $\theta_{\text{glial}}$：
$$\theta_{t} = \theta_{t-1} + K_p e_t + K_i \int e_t dt + K_d \frac{de_t}{dt}$$ 
這使得系統在 80% 以上的高斯雜訊干擾下，仍能透過調整神經元敏感度維持 **1-step** 的直覺辨識能力。

---

## 📊 4. 實驗數據與分析 (Experimental Evaluation)

### 4.1 基準測試：MNIST 分類任務
我們在 **Intel Arc B580 (Discrete GPU)** 上對 4096 隱層維度的 PNB-X 模型進行了評測。

| 模型架構 | 權重位元 | 記憶體佔用 | 辨識準確率 (Top-1) | 思考步數 (30% 雜訊) |
| :--- | :---: | :--- | :--- | :--- |
| Baseline MLP | 32-bit | ~100.5 MB | 98.60% | 1 (固定) |
| 傳統 SNN (LIF) | 32-bit | ~100.5 MB | 98.50% | 50 - 100 (迭代) |
| **PNB-X (Ours)** | **1.58-bit** | **~12.1 MB** | **99.15%** | **1 (直覺模式)** |

### 4.2 魯棒性分析 (Noise Resilience)
實驗顯示，當環境雜訊（Gaussian Noise）增加時：
*   **無 CLS 狀態**：系統信心顯著下降，且在極端雜訊 (SNR < 0dB) 下需要更多迭代步數才能達成閾值。
*   **CLS 增強模式**：透過海馬體的一次性特徵繫結，系統能將原本需要數十步的「猶豫期」縮短至單步反應，信心值提升幅度約 **10% ~ 15%**。

---

## 💻 5. 硬體層級優化 (Hardware-Aware Optimization)

PNB 使用 **SYCL (Intel oneAPI)** 撰寫，實作了多項關鍵的底層優化：
1.  **XMX 三元卷積核心**：針對 Intel GPU 上的矩陣擴展單元（Xe Matrix Extensions）開發了自定義矩陣乘法核，專門處理 $\{-1, 0, 1\}$ 權重對映。
2.  **向量化前綴和 (Vectorized Prefix Sum)**：優化了 SSM 中的線性掃描操作，利用 `sub_group` 的 Shuffle 操作實現了超低延遲的狀態更新。
3.  **零開銷視圖 (Zero-overhead Views)**：廣泛採用 `std::experimental::mdspan` 代替傳統的多維指針，在確保型別安全的同時消除了執行時的索引計算開銷。

---

## 🛠️ 6. 開發者指南 (Getting Started)

### 6.1 環境配置
*   **Compiler**: `icpx` (Intel oneAPI DPC++/C++ Compiler 2025.x)
*   **Runtime**: Level Zero 或 OpenCL 驅動
*   **Standards**: C++2b (C++23)

### 6.2 編譯與執行
```bash
# 執行綜合整合測試
icpx -fsycl -std=c++2b -Iinclude -Imdspan/include test/test_cls_mnist_integration.cpp -O3 -o pnb_test
./pnb_test
```

### 6.3 訓練流程
```bash
# 基於代理梯度的 MNIST 訓練
icpx -fsycl -std=c++2b -Iinclude -Imdspan/include test/train_mnist_surrogate.cpp -O3 -o pnb_train
./pnb_train
```

---

## 📂 7. 專案目錄結構 (Project Structure)

```text
PNB/
├── include/neurobit/
│   ├── core/           # 核心動力學、Bit-packing、硬體抽象層
│   ├── layers/         # BitNet b1.58、SSM-Scan、代理梯度激活層
│   └── components/     # Glial (膠質調節)、Hippocampus (記憶轉移)
├── test/               # 整合測試、馬拉松推論驗證、動態場景切換測試
├── data/               # (Local only) MNIST 原始二進位數據集
├── exports/            # (Local only) 訓練好的 .bin 與 .onnx 模型
└── LICENSE             # Apache License 2.0
```

---

## 📑 8. 引用格式 (BibTeX)

若本專案對您的研究有所啟發，請按以下格式引用：

```bibtex
@software{pnb2025,
  author = {Project Neuro-Bit Contributors},
  title = {PNB-X: A High-Performance Quantized Brain-Inspired Inference Engine},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/matrix/PNB}}
}
```

---

## 📚 9. 參考文獻 (References)

1.  **BitNet b1.58**: *Ma et al. (2024)*. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits". [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
2.  **Mamba (Selective SSM)**: *Gu & Dao (2023)*. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces". [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
3.  **Surrogate Gradients in SNN**: *Neftci et al. (2019)*. "Surrogate Gradient Learning in Spiking Neural Networks". [Signal Processing Magazine](https://ieeexplore.ieee.org/document/8830399)
4.  **Complementary Learning Systems**: *McClelland et al. (1995)*. "Why there are complementary learning systems...". [Psychological Review](https://psycnet.apa.org/record/1995-39148-001)
5.  **Homeostatic Plasticity**: *Turrigiano (2012)*. "The self-tuning neuron: synaptic scaling...". [Cell Press](https://www.cell.com/neuron/fulltext/S0896-6273(12)00834-1)

---
**Project Neuro-Bit** - *探索量化神經元的極限與仿生智慧的未來。*