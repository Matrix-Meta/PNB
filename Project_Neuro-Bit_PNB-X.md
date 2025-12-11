# Project Neuro-Bit PNB-X  

非 Transformer 類腦 BitNet 架構藍圖  
（SNN + micro-RNN + SSM + Fast Weights，全狀態流式架構）

目標硬體：Intel Xeon CPU + Intel Arc / NVIDIA CUDA GPU  

核心目標：  

1. 完全拋棄 Transformer / KV cache，不在「時間軸」上做 self-attention。  
2. 以「有限維全局狀態 State_t」承載所有歷史資訊：  
   \[
   \text{State}_t = \{(s_t^l,u_t^l,v_t^l,\dots)\}_{l=1}^L
   \]  
3. 微觀層使用 SNN（膜電位＋脈衝）提供稀疏與事件驅動；  
   次微觀層使用小 RNN + fast weights 存局部可塑性；  
   長程依賴交給 SSM（State-Space Model）；  
   水平傳播使用「局部神經元訊息傳遞（小範圍自注意／Message Passing）」而不是 token-wise self-attention。  
4. 透過 Glial Controller + CLS（海馬體/新皮層）實作快慢記憶與自適應內部推理迴圈。

---

## 0. 為何可以「不用 Transformer」仍處理長序列？

### 0.1 基本設定：因果序列模型

考慮一般自回歸序列模型：  
給定輸入序列 \(x_{1:t} = (x_1,\dots,x_t)\)，輸出分佈：

\[
p(y_t \mid x_{1:t}) = F_t(x_{1:t})
\]

其中 \(F_t\) 是某個因果映射：只依賴過去，不依賴未來。

Transformer 做的是：  

- 每個時間步 t 都顯式持有所有過去 token 的表示 \(\{h_1,\dots,h_t\}\)，  
- 用 self-attention 計算 \(h_t' = \text{Attn}(h_t,\{h_1,\dots,h_t\})\)。  
- 為避免重算，使用 KV cache 存 \(\{K_1..K_t,V_1..V_t\}\)。

我們現在要做的是：  

- 不再存整條歷史，而是只存一個有限維狀態 \(S_t\)，  
- 用「狀態遞推」來吸收歷史資訊。

---

### 0.2 命題 1：任何有限記憶的因果轉換，都可由有限維狀態模型表示

假設任務有「有限記憶長度」M，即對所有 t：

\[
F_t(x_{1:t}) = f(x_{t-M+1:t})
\]

對於 t < M 可 padding。  

構造一個「顯式狀態」：

\[
S_t := (x_{t-M+1},\dots,x_t)
\]

那麼：  

1. 狀態更新是有限維的：
   \[
   S_{t+1} = U(S_t,x_{t+1})
   \]
   實際上就是把 window 往右平移一格再塞 x_{t+1}。  

2. 輸出是狀態的函數：
   \[
   y_t = G(S_t)
   \]

→ 任意有限記憶的因果序列轉換都可以寫成「有限維狀態 + 狀態遞推」的形式。

實際上，多數 NLU/NLG 任務不會真的是嚴格有限記憶，但實務上常用「有效記憶長度」的概念（超過某長度貢獻很小），所以可以視為上述的近似。

---

### 0.3 命題 2：RNN / SSM + 非線性可以近似任意狀態遞推

RNN universal approximation / SSM + MLP 的結果告訴我們（只給直覺版本）：

- 對任意連續函數 \(U(S,x)\)、\(G(S)\)，  
  存在某個足夠大的 RNN / SSM + MLP，可以在任意精度 ε 下近似它。  
- 也就是說：  
  \[
  S_{t+1} \approx \hat U_\theta(S_t,x_t), \quad y_t \approx \hat G_\theta(S_t)
  \]

因此，只要：

1. 我們維持一個固定維度的 State_t；  
2. 用 RNN + SSM + SNN + fast weights 組合實現一個夠 expressive 的 \(\hat U_\theta\)、\(\hat G_\theta\)；  

就可以「不用 Transformer」逼近同樣的因果序列映射。

---

### 0.4 命題 3：在這套設計下，記憶使用與序列長度無關（無 KV）

在本架構中，推理時每一步只持有當前狀態：

\[
\text{State}_t = \{(s_t^l,u_t^l,v_t^l,\dots)\}_{l=1}^L
\]

更新規則：

\[
\text{State}_{t+1} = F_\theta(\text{State}_t, x_t)
\]

- 這個 State 含所有層的 SSM、RNN、SNN、fast-weight 狀態。  
- 它的維度與「模型大小」成正比，但與序列長度 t 無關。  

因此：

> 命題：PNB-X 在推理階段的常駐記憶量是 O(#layers × state_dim)，與序列長度 L 無關，不需要 per-token KV cache。

Transformer 必須儲存所有 token 的 key/value；PNB-X 只保持「一份狀態」，這是整個「不需要 KV cache」的理論基礎。

---

## 1. 架構總覽：PNB-X 模組

### 1.1 全局狀態結構

對 batch size = B，模型有 L 層，每層有 N_l 個「神經元 / microcircuit」：

\[
\text{State}_t = \Big\{\big(s_t^l\in\mathbb{R}^{B\times d_{ssm}^l},\ u_t^l\in\mathbb{R}^{B\times N_l\times d_{rnn}^l},\ v_t^l\in\mathbb{R}^{B\times N_l},\ z_t^l\in\mathbb{R}^{B\times N_l}\big)\Big\}_{l=1}^L
\]

- \(s_t^l\)：第 l 層 SSM 長程狀態  
- \(u_t^l\)：第 l 層微觀 RNN 狀態  
- \(v_t^l\)：膜電位  
- \(z_t^l\)：脈衝（spike）指示（0/1 或連續近似）

### 1.2 單步更新總形式

對每個時間步 t，輸入 token embedding \(x_t \in \mathbb{R}^{B \times d_\text{model}}\)：

\[
(x_{t}^{1}, \text{State}_{t}^{1}) = \text{InputLayer}(x_t, \text{State}_t^{1})
\]
\[
(x_{t}^{l+1}, \text{State}_{t}^{l+1}) = \text{BitBrain}^l(x_{t}^l, \text{State}_t^l), \quad l=1..L
\]

最終：

\[
h_t = \text{Readout}(x_{t}^{L}, \text{State}_t^L)
\]
\[
\text{logits}_t = W_o h_t + b_o
\]

---

## 2. 單層 BitBrain：SSM + micro-RNN + SNN + 局部訊息傳遞 + Fast Weights

對第 l 層，在時間步 t，我們定義：

- 輸入：\(x_t^l\)  
- 輸出：\(x_t^{l+1}\)  
- 層狀態：\(s_t^l, u_t^l, v_t^l, z_t^l, M_t^l\)（其中 \(M_t^l\) 是 fast-weight matrix 或其壓縮表示）

### 2.1 SSM 長程動態

採用離散線性狀態方程：

\[
s_{t+1}^l = A^l s_t^l + B^l x_t^l
\]
\[
y_{\text{ssm},t+1}^l = C^l s_{t+1}^l + D^l x_t^l
\]

其中 \(A^l,B^l,C^l,D^l\) 為可學參數（可採 block-diag 結構以利實作）。

**穩定性條件（簡略）：**  
若希望序列長度→∞ 時仍不發散，可讓 \(A^l\) 的譜半徑 \(\rho(A^l) < 1\)，例如：

- 參數化為 \(A^l = \tanh(\tilde A^l)\) 的對角或 block-diag 矩陣。  

這樣可確保：

\[
\|s_{t}^l\| \leq c + \rho^t \|s_0^l\|
\]

---

### 2.2 局部神經元訊息傳遞（小範圍自注意／Message Passing）

對第 l 層有 N_l 個「神經元結點」或小 microcircuit，組成一個圖 G^l，其鄰居集合 \(\mathcal{N}(i)\) 限於局部。

定義上一時刻的「可見向量」：

\[
h_i^{l,t} = \text{concat}[u_i^{l,t}, v_i^{l,t}, y_{\text{ssm},t}^l]
\]

對每個 i，我們做局部注意／訊息傳遞：

1. 計算鄰域 attention score：

\[
e_{ij}^{l,t} = \psi\big(q(h_i^{l,t}), k(h_j^{l,t})\big), \quad j\in\mathcal{N}(i)
\]

2. 正規化：

\[
\alpha_{ij}^{l,t} = \frac{\exp(e_{ij}^{l,t})}{\sum_{k\in\mathcal{N}(i)} \exp(e_{ij}^{l,t})}
\]

3. 聚合鄰域訊息：

\[
c_i^{l,t} = \sum_{j\in\mathcal{N}(i)} \alpha_{ij}^{l,t} \, v(h_j^{l,t})
\]

這裡的「注意力」只在神經元圖上運作，而且是同一時間步 t 的神經元之間的互動；  
**不涉及 token 歷史，不需要 KV cache。**

---

### 2.3 micro-RNN 更新

對每個神經元 i 的小 RNN 狀態 \(u_i^{l,t}\)，可用簡化 GRU / RNN：

\[
u_i^{l,t+1} = \phi\big(W_u^l u_i^{l,t} + U_u^l c_i^{l,t} + V_u^l y_{\text{ssm},t+1}^l\big)
\]

- \(\phi\) 可為 tanh / ReLU / GELU。  
- 這一步把「長程 SSM 訊號」+「局部 graph 訊息」整合到新的內部狀態。

---

### 2.4 微觀 SNN：膜電位與脈衝

每個神經元 i 有：

- 膜電位 \(v_i^{l,t}\)  
- 脈衝輸出 \(z_i^{l,t}\)  

更新規則：

1. 膜電位更新（含 OU 噪聲）：

\[
v_i^{l,t+1} = \alpha^l v_i^{l,t} + W_v^l u_i^{l,t+1} + b_v^l + g_{\text{noise}}^l X_i^{l,t+1}
\]

其中 \(X_i^{l,t}\) 為 OU 噪聲（見下一節），\(\alpha^l\in(0,1)\) 是遺忘係數。

2. 脈衝放電（用平滑階躍近似）：

\[
z_i^{l,t+1} = \sigma_{\text{spike}}(v_i^{l,t+1} - \theta^l)
\]

\(\sigma_{\text{spike}}\) 可用：

\[
\sigma_{\text{spike}}(x) \approx \frac{1}{1 + \exp(-k x)}
\]

在推理時可將其硬量化為：

\[
\tilde z_i^{l,t+1} = \mathbb{1}(v_i^{l,t+1} > \theta^l)
\]

3. 稀疏輸出：

\[
\tilde h_i^{l,t+1} = z_i^{l,t+1} \cdot f(u_i^{l,t+1}, v_i^{l,t+1})
\]

- 若 \(z_i\approx 0\)，該神經元在此步幾乎不輸出訊號 → 時間稀疏。

---

### 2.5 Fast Weights：局部可塑性／短期記憶

在神經元 i 與鄰居 j 之間設定 fast-weight 矩陣 \(M_{ij}^{l,t}\)（可壓縮成低秩或共享結構）。

更新規則（Hebbian-like）：

\[
M_{ij}^{l,t+1} = \lambda^l M_{ij}^{l,t} + \eta^l \, h_i^{l,t} h_j^{l,t\top}
\]

- \(\lambda^l \in (0,1)\)：遺忘係數；  
- \(\eta^l > 0\)：fast-weight 學習率。  

在下一步局部訊息傳遞時，可額外加上 fast-weight 項：

\[
c_i^{l,t} = \sum_{j\in\mathcal{N}(i)} \alpha_{ij}^{l,t} \, v(h_j^{l,t}) + \beta^l \sum_{j\in\mathcal{N}(i)} M_{ij}^{l,t} h_j^{l,t}
\]

→ 這等價於把最近出現過的 pattern encode 成一個小 Hopfield-like 記憶。

---

### 2.6 層輸出與 BitNet 量化

將各子模組輸出組合成層輸出：

\[
h_i^{l,t+1} = W_h^l \big[y_{\text{ssm},t+1}^l, u_i^{l,t+1}, v_i^{l,t+1}, z_i^{l,t+1}\big] + b_h^l
\]

再經過 BitNet 量化線性層：

- 權重三元量化 \(W_{fp}^l \to W_{q}^l \in \{-1,0,1\}\)。  
- 前向使用整數 matmul；反向用 STE。

\[
x_t^{l+1} = \text{BitLinear}^l(h^{l,t+1})
\]

---

## 3. OU 噪聲作為隨機基底（與 SNN 融合）

OU 過程離散化（每神經元一維）：

\[
X_{t+1} = X_t + \theta(\mu - X_t)\Delta t + \sigma\sqrt{\Delta t}\,\epsilon_t
\]
\[
\epsilon_t \sim \mathcal{N}(0,1)
\]

- 訓練：\(\sigma\) 較大，鼓勵探索與正則；  
- 推理：\(\sigma \to 0\) 或極小。  

在膜電位更新式中：

\[
v_i^{l,t+1} = \dots + g_{\text{noise}}^l X_i^{l,t+1}
\]

- \(g_{\text{noise}}^l\) 由 Glial 控制。  
- OU 的「回復特性」讓噪聲不會長期飄移。

---

## 4. Glial Controller：全域調節與內部推理控制

### 4.1 觀測量

對每個時間步 t，我們從 logits / 狀態中提取：

- cross-entropy / negative log-likelihood：
  \[
  \text{NLL}_t = -\log p_\theta(y_t \mid x_{1:t})
  \]
- entropy：
  \[
  H_t = -\sum_i p_i \log p_i
  \]
- 各層放電比例（活躍度）：
  \[
  \rho_t^l = \frac{1}{B N_l}\sum_{b,i} \mathbb{1}(z_{b,i}^{l,t} > \tau)
  \]
- fast-weights 更新量 \(\|M_{ij}^{l,t+1}-M_{ij}^{l,t}\|\) 等。

我們對這些量做指數平滑：

\[
\hat H_t = \alpha \hat H_{t-1} + (1-\alpha) H_t
\]

---

### 4.2 控制輸出

Glial 輸出：

- LayerNorm 增益 \(\alpha_\text{LN}^l\)、偏移 \(\beta_\text{LN}^l\)  
- 噪聲 gate \(g_{\text{noise}}^l\)  
- 放電閾值調整 \(\Delta\theta^l\)  
- fast-weights 學習率 \(\eta^l\)  
- 內部 reasoning 最大深度 \(max\_think\_depth\)  
- CLS read/write flag：read_enable, write_enable  

可以先用 rule-based：

```pseudo
if H_t > HIGH_ENTROPY_TH:
    max_think_depth = 3
    g_noise = 1.0
    read_enable = True
    write_enable = True
elif H_t < LOW_ENTROPY_TH:
    max_think_depth = 0
    g_noise = 0.2
    read_enable = False
    write_enable = False
else:
    max_think_depth = 1
    g_noise = 0.5
    read_enable = True
    write_enable = False
```

也可以用 MLP/RNN 把過去 K 步統計餵進去做 learnable policy。

---

## 5. CLS：快慢記憶（海馬體 + 新皮層）在無 Transformer 模型中的角色

### 5.1 海馬體（快記憶）

對每個「高驚奇度」時間步 t，我們寫入一筆記憶：

- key：
  \[
  k_t = f_k(h_t, \text{State}_t)
  \]
- value：
  \[
  v_t = f_v(h_t, y_t, \text{State}_t)
  \]

寫入條件：

- \(\text{NLL}_t > \mu_\text{NLL} + \delta\) 且 Glial.write_enable = True。

讀取時（訓練/推理）：

1. 用當前 query：
   \[
   q_t = f_q(h_t,\text{State}_t)
   \]
2. 做 KNN：
   \[
   \{(k_{i},v_{i})\}_{i \in \mathcal{I}_t} = \text{TopK}( \text{sim}(q_t,k_j) )
   \]
3. 合成 memory context：
   \[
   m_t = \sum_{i \in \mathcal{I}_t} \alpha_i v_i,\quad \alpha_i = \text{softmax}_i(\text{sim}(q_t,k_i))
   \]
4. 融入狀態：
   \[
   \tilde h_t = h_t + W_m m_t
   \]

---

### 5.2 新皮層（慢記憶）

新皮層 = PNB-X 主模型權重；  
Consolidation 重播海馬體樣本，在不需要 KV 的前提下重放序列、更新權重，並加上正交化正則：

\[
L_\text{ortho} = \sum_{\text{blocks }B} \| B^\top B - I \|_F^2
\]

總 loss：

\[
L = L_\text{task} + \lambda_\text{ortho} L_\text{ortho}
\]

---

## 6. 內部自推理機：在固定狀態上多步反芻

### 6.1 思路

在每個 token 輸出前：

1. 外層先根據當前 State_t 給一個 logits_outer。  
2. Glial 觀察 entropy / margin，若不確定 → 啟動 internal loop。  
3. internal loop 多次在「同一時間步」更新 State（不吃新 token），讓 SSM/RNN/SNN/CLS 持續交互，直到不確定度降低。  
4. 最後用更新後的 State 產生 logits_final。

這等價於對「當前狀態下的決策」做一個小型迭代 solver。

### 6.2 簡略收斂直覺

若 internal loop 對 State 的更新近似一個「壓縮映射」：

\[
\|\Phi(\text{State}) - \Phi(\text{State}')\| \leq c \|\text{State} - \text{State}'\|,\quad c<1
\]

且 entropy 隨 loop 單調下降（在多數樣本上）：

\[
H^{(k+1)} \leq H^{(k)} - \epsilon
\]

則有限步數 K 內，可以把不確定度降到某個目標範圍。實務上我們不做嚴格收斂保證，而是：

- 用訓練 + RL/meta-learning 讓 Glial 學會「何時多跑幾步」；  
- 限制 max_think_depth，避免發散迴圈。

---

### 6.3 internal_reasoning_loop 偽代碼

```pseudo
fn internal_reasoning_loop(state, h_t, glial_state):
    steps = 0
    thought_state = mark_reasoning_mode(state)   # 在狀態中加入 "reasoning" 標記

    while steps < glial_state.max_think_depth:
        # 1. 不吃新 token，只用 "空輸入" 或特定 THINK embedding
        think_input = get_internal_input()  # e.g. learned vector or zeros

        h_inner, thought_state, inner_stats = bitbrain_forward(
            think_input, thought_state, mode="infer_internal"
        )

        # 2. Glial 更新對不確定度/能量的估計
        glial_state = glial.update(inner_stats)

        # 3. OPTIONAL: 更新內部思考 buffer（僅作為訓練對齊用，不輸出）
        update_thought_trace(thought_state, h_inner)

        # 4. 判斷是否已經足夠確定
        if glial_state.confidence_high_enough:
            break

        steps += 1

    fused_state = fuse_state(state, thought_state)
    fused_h = fuse_hidden(h_t, h_inner)
    return fused_state, fused_h, glial_state
```

---

## 7. 訓練目標與梯度傳播

### 7.1 基本 LM loss

對 batch size B、長度 T 的序列：

\[
L_\text{task} = -\frac{1}{BT}\sum_{b=1}^B\sum_{t=1}^T \log p_\theta(y_{b,t}\mid x_{b,1:t})
\]

### 7.2 SNN 的近似梯度（surrogate gradient）

對 spike 函數 \(z = \mathbb{1}(v > \theta)\) ，在反向時用平滑近似：

- forward: 硬階躍或夾逼到 [0,1]  
- backward:  
  \[
  \frac{\partial z}{\partial v} \approx \sigma'(v-\theta) = \frac{k e^{-k(v-\theta)}}{\big(1+e^{-k(v-\theta)}\big)^2}
  \]

或其他 piecewise linear 近似。

### 7.3 時間展開與 BPTT

訓練時對 T 步展開，對所有 \(s_t^l, u_t^l, v_t^l, M_t^l\) 做反向傳播（可用 truncated BPTT）。

---

## 8. 偽代碼：實作骨架

### 8.1 資料結構

```pseudo
struct LayerState {
    s   # SSM state, shape [B, d_ssm]
    u   # micro-RNN state, shape [B, N, d_rnn]
    v   # membrane potential, shape [B, N]
    z   # spike output, shape [B, N]
    M   # fast weights (compressed), e.g. [N, low_rank, N]
    ou  # OU noise state, shape [B, N] or [B, d_model]
}

struct GlobalState {
    layers: [LayerState; L]
    glial_state
}
```

### 8.2 單層前向

```pseudo
fn bitbrain_layer_forward(x, layer_state, glial_ctrl, mode):
    # 1. SSM 更新
    s_next   = A @ layer_state.s + B @ x
    y_ssm    = C @ s_next + D @ x

    # 2. 建立神經元可見向量
    h_prev = concat(u = layer_state.u,
                    v = layer_state.v_unsqueezed,
                    y = broadcast(y_ssm))

    # 3. 局部訊息傳遞（不跨時間）
    c = local_message_passing(h_prev, layer_state.M)  # 含 fast weights

    # 4. micro-RNN 更新
    u_next = phi(W_u @ layer_state.u + U_u @ c + V_u @ y_ssm_broadcast)

    # 5. OU 噪聲 + 膜電位 + spike
    step(layer_state.ou, eps_std_normal())  # OU step
    noise = current(layer_state.ou)

    v_next = alpha * layer_state.v + W_v @ u_next + b_v + glial_ctrl.g_noise * noise
    z_next = spike_fn(v_next - theta_adjusted(glial_ctrl))

    # 6. 層輸出 + BitNet 量化線性
    h_next = W_h @ concat(y_ssm_broadcast, u_next, v_next, z_next) + b_h
    x_next = bitlinear_forward(h_next, ...)

    # 7. 更新 fast weights
    M_next = update_fast_weights(layer_state.M, h_prev, h_next, glial_ctrl.eta_fast)

    new_state = LayerState(s_next, u_next, v_next, z_next, M_next, layer_state.ou)
    layer_stats = collect_layer_stats(...)
    return x_next, new_state, layer_stats
```

### 8.3 單步整模型前向

```pseudo
fn bitbrain_forward_step(x_t, global_state, mode):
    h = x_t
    layer_stats_all = []

    for l in range(L):
        h, global_state.layers[l], layer_stats = bitbrain_layer_forward(
            h,
            global_state.layers[l],
            global_state.glial_state,
            mode
        )
        layer_stats_all.append(layer_stats)

    logits = lm_head(h)
    return logits, h, global_state, layer_stats_all
```

### 8.4 訓練主迴圈（無 Transformer / KV）

```pseudo
for each training_step:
    batch = sample_batch()  # (B, T)
    global_state = init_global_state(B)
    all_logits = []

    for t in range(T):
        x_t = embed(batch.inputs[:, t])

        logits_t, h_t, global_state, layer_stats = bitbrain_forward_step(
            x_t, global_state, mode="train"
        )
        all_logits.append(logits_t)

        # Glial 更新
        stats_t = compute_stats(logits_t, batch.targets[:, t], layer_stats)
        global_state.glial_state = glial.update(stats_t, global_state.glial_state)

        # 海馬體寫入
        if global_state.glial_state.write_enable and is_high_surprise(stats_t):
            hippocampus.write(global_state, h_t, logits_t, stats_t)

    logits = stack_time(all_logits)
    loss_task = cross_entropy(logits, batch.targets)
    loss_ortho = neocortex.orthogonal_regularization()
    loss = loss_task + lambda_ortho * loss_ortho

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if training_step % CONSOLIDATION_INTERVAL == 0:
        consolidate(neocortex, hippocampus)
```

### 8.5 推理＋內部自推理

```pseudo
fn generate_streaming(prompt, max_tokens):
    global_state = init_global_state(1)
    output = []
    context = encode_prompt(prompt)

    for t in range(max_tokens):
        x_t = embed(get_last_token_or_bos(output, context))

        # 外層一步
        logits_outer, h_outer, global_state, layer_stats = bitbrain_forward_step(
            x_t, global_state, mode="infer_outer"
        )
        stats_outer = compute_stats_from_logits(logits_outer)
        global_state.glial_state = glial.update(stats_outer, global_state.glial_state)

        if global_state.glial_state.need_reasoning:
            global_state, h_final, global_state.glial_state = internal_reasoning_loop(
                global_state, h_outer, global_state.glial_state
            )
            logits_final = lm_head(h_final)
        else:
            logits_final = logits_outer

        token = sample_token(logits_final, global_state.glial_state.sampling_strategy)
        output.append(token)

        if token in ["<EOS>", "<STOP>"]:
            break

    return detokenize(output)
```

---

## 9. Roadmap（實作順序建議）

1. **Phase 0：BitNet + 單層 RNN baseline**  
   - 不用 Transformer，只用 BitNet 線性層 + 簡單 RNN state。  
   - 確認小語料上可收斂。

2. **Phase 1：加入 SSM 長程通道**  
   - 在每層加入 SSM 狀態 \(s_t^l\)，替代原本會想用 self-attention 記長程資訊的地方。

3. **Phase 2：導入 SNN 膜電位與 OU 噪聲**  
   - 將普通激活替換為膜電位 + spike gate。  
   - 加入 OU 噪聲並由超參數控制強度。

4. **Phase 3：加入局部神經元訊息傳遞 + fast weights**  
   - 在層內建立神經元 graph，實作 local message passing。  
   - 使 fast weights 作為短期可塑性，驗證 pattern recall 能力。

5. **Phase 4：實作 Glial Controller + CLS**  
   - 整合驚奇度、稀疏度監控 → 控制噪聲、閾值、稀疏度。  
   - 實作海馬體快記憶與新皮層 consolidation。

6. **Phase 5：加入內部自推理機**  
   - 實作 internal_reasoning_loop（固定步數），再加入 Glial 動態控制。  
   - 用「教師顯式 CoT → 學生隱式狀態軌跡對齊」蒸餾推理行為。
