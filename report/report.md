# 影像處理作業二：影像銳化

## 一、技術說明｜作業目的與程式流程

### 1.1 作業目的與環境

對六張 512 × 420 的 8-bit 灰階 PNG 使用以下五種銳化方法。

1. Laplacian Operator（拉普拉斯算子）

2. Unsharp Masking（反遮罩）

3. High-boost Filtering（高提升濾波）

4. Homomorphic Filtering（同態濾波）

5. Combination of Two or More Image Enhancement Techniques in the Frequency Domain（在頻域結合兩種以上的影像增強技術）

本作業從影像處理的角度比較邊緣銳利度、局部對比、細節與顆粒、灰階分布及飽和比例，並以頻率響應說明頻域濾波器對不同尺度的影響。輸入的灰階值為 0 至 255，分別代表黑與白。

開發環境為 Windows 11、Python 3.12 和 uv。NumPy 提供浮點與複數陣列運算，Pillow 解碼與儲存 PNG、排版及縮放顯示，Matplotlib 繪製頻譜，Tkinter 顯示視窗。pytest 檢驗手算公式及整合流程，PyInstaller 封裝 Windows 執行檔。專案以 ReportLab 由 Markdown 產生 PDF。

### 1.2 架構與程式執行流程

```
CLI / Tkinter
  -> validate PNGs and load JSON parameters
  -> float64 pixels
     |-> Laplacian convolution -> f - alpha * L
     |-> separable Gaussian -> detail -> unsharp and high-boost
     |-> log1p -> mirrored extension -> 2-D FFT
                 |-> H_HOMO -> inverse FFT
                 |-> H_HOMO * H_HB -> inverse FFT
                         -> crop -> expm1
  -> limit to [0,255], round to uint8
  -> write images, figures, and results.json
```

1. `cli.py` 解析輸入、輸出、方法與設定。批次開始前，`pipeline.py` 比對 2-1.png 至 2-6.png 的檔名，驗證 PNG 容器、原生 8-bit 灰階與長寬，再逐一解碼。原始位元組與陣列分別保留，輸出目錄不得和輸入重疊。

2. `config.py` 合併共用預設值和每張圖的覆寫欄位，檢查範圍及資料型別。`algorithms.py` 將影像提升到 float64，兩種空間高斯方法共用細節遮罩，兩種頻域方法共用正向 FFT。各分支最後限幅及取整。

3. `pipeline.py` 在暫存目錄寫入原圖副本、增強圖和圖表，計算數據及 SHA-256，再發布檔案與結果清單。原圖直接複製位元組。各輸出灰階圖保持 512 × 420、8-bit 及像素座標。

4. `gui.py` 在主視窗的六格比較圖下方直接顯示對數影像頻譜與三條濾波增益曲線。下拉選單、前後按鈕及左右鍵會一起更新影像、圖表與參數。小視窗可垂直捲動，雙擊影像或圖表可用原始解析度查看。Regenerate preview 在背景產生增強圖和對應圖表，再依檔名一起更新預覽，保留已儲存的檔案。

5. `build_report.py` 驗證 Markdown、設定對應的結果和圖表後生成 PDF。PyInstaller 內附原圖、結果及設定，封裝腳本核對本文檔及執行檔的來源指紋與 ZIP 內容。

<!-- page -->
## 一、技術說明｜空間域方法

### 1.3 手動二維捲積與邊界

對奇數尺寸核 K，y 為列、x 為行內橫向座標，輸出為 `B(y,x) = Σ(i,j) K(i,j) · f(y-i,x-j)`。程式把核心沿兩軸反轉，再逐一累加權重乘以對應影像切片。即使核心不對稱，方向也符合捲積定義。反射邊界包含端點，避免零填補造成的黑邊。對長度 N 的索引 j，令 `t = j mod (2N)`，映射為 `min(t, 2N-1-t)`。此規則也適用於單列或單行影像。

```
for (y, x), weight in np.ndenumerate(weights[::-1, ::-1]):
    if weight:
        filtered += weight * extended[y:y+height, x:x+width]
```

### 1.4 Laplacian Operator（拉普拉斯算子）

採四鄰域核 `[[0,1,0], [1,-4,1], [0,1,0]]`。對稱延伸後，以 `L = f(y+1,x) + f(y-1,x) + f(y,x+1) + f(y,x-1) - 4f(y,x)` 估計二階差分。銳化公式為 `g = f - αL`。對亮中心 L 為負，因此減去 L 會提高該中心的灰階。α 越大，反差越大，也越可能加強噪聲和造成邊緣過衝。程式支援八鄰域實驗，本次六張皆用四鄰域。

### 1.5 Unsharp Masking（反遮罩）

以 `w(t) = exp[-0.5 · (t/σ)²]` 計算一維權重並除以 Σw。二維高斯核為兩組正規化向量的外積，總和為 1。利用可分離性，先對水平方向再對垂直方向捲積，得到 `B = G * f`。細節遮罩為 `M = f - B`，輸出為 `g = f + M`，增益固定為 1。σ 控制平滑範圍，核尺寸限制有效支撐域。對 H × W 影像與 m × m 核，兩次一維捲積約需 O(HWm) 運算。

### 1.6 High-boost Filtering（高提升濾波）

輸出為 `g = f + kM`，其中 k > 1。Unsharp Masking（反遮罩）是同一公式中 k = 1 的情形。本實驗對同一張圖以同一組核尺寸與 σ 計算 M，藉由不同遮罩增益比較兩方法。低對比細節因 k 提高而更顯眼，細粒、非結構性噪聲與邊緣暈圈也可能同步變強。三種空間域方法均使用 float64 運算，最後對 [0,255] 限幅，以 `floor(value + 0.5)` 取整並轉為 uint8。

<!-- page -->
## 一、技術說明｜手動 FFT 與頻率座標

### 1.7 Radix-2 FFT 與 IFFT

一維正向 DFT 定義為 `X[k] = Σ(n=0…N-1) x[n] exp(-j 2πkn/N)`。Cooley–Tukey 將偶數與奇數索引分解成長度 N/2 的轉換，使用 `W = exp(-j 2πk/N)` 結合：同一對輸出分別為 `E + W·O` 與 `E - W·O`。本程式用位元反轉實作迭代式 decimation in time，蝶形分組長度由 2 逐次加倍。各軸長的位元反轉與 cos/sin 旋轉因子有快取，採複數陣列批次處理同一階段的區塊。

```
result = contiguous(values[..., bit_reversed_indices])
for stage_twiddles in plan:
    groups = reshape_by_stage(result)
    even = copy(first_half(groups))
    odd  = second_half(groups) * stage_twiddles
    first_half(groups)  = even + odd
    second_half(groups) = even - odd
```

逆轉換改用共軛旋轉因子，最後除以該軸長度。二維正向與逆向都先沿各列做一維轉換，再透過轉置沿各行做一維轉換，二維逆轉換因此總共除以 H × W。時間複雜度為 O(HW(log H + log W))，H、W 指擴展後的尺寸。小型複數陣列以直接計算 DFT 的測試獨立核對 FFT 結果。

### 1.8 對稱延伸、頻率索引與裁切

原圖陣列為 420 列 × 512 行內像素。對 `log1p(f)` 於垂直軸兩側各反射 46 列，先得到 512 × 512；水平方向原已達 512 點。這個方形陣列沿兩軸各反射一份，成為 1024 × 1024 的週期偶對稱資料。反射接縫及週期首尾的相鄰像素值一致，可減少由左右或上下直接相接產生的不連續；影像內容自身的明暗邊界仍會影響濾波結果。

頻域計算採未移位座標。偶數軸長 N 時，索引 0 至 N/2-1 除以 N 得非負頻率，N/2 至 N-1 先減 N 再除以 N。距離平方為 `D² = u² + v²`，u、v 的單位為 cycles/pixel，因此截止頻率與實際像素尺度一致。濾波時 DC 在 [0,0]。繪圖才用 `(index - N//2) mod N` 重排，將 DC 移到中央。

兩個濾波器的增益皆為實數且呈偶對稱，保留實數訊號的共軛對稱性。IFFT 後取實部並裁切原位置 `[46:466, 0:512]`。其後套用指數回復與量化；運算與儲存都維持原始方向。

<!-- page -->
## 一、技術說明｜頻域方法

### 1.9 Homomorphic Filtering（同態濾波）

照明與反射模型為 `f(x,y) = i(x,y) · r(x,y)`。對數可將正值乘積轉成加法。自然影像常把平緩照明視作偏低頻，局部結構視作帶有較多高頻的成分，兩者的頻帶仍會重疊。實作使用 `z = log1p(f)` 以處理 0 像素，對乘法模型的對數分離是近似。在本作業中，此模型用於實驗調整局部對比與大尺度亮暗。

對 z 做 FFT 得 Z。逐 bin 建立濾波器，`H = γL + (γH - γL)[1 - exp(-c · D²/D0²)]`。γL 控制接近 DC 的低頻衰減，γH 控制高頻放大量，D0 為轉換尺度，c 控制過渡速度。為保留擴展對數圖的平均值，本作業將 H[0,0] 改為 1，其餘低頻照公式衰減。對數平均保留後，指數域的算術平均仍可變動。

將 `H · Z` 作 IFFT，取實部並裁切。使用 `expm1` 回到原單位。轉換前限制對數值至 [0, log(256)]，這與指數後限制灰階至 [0,255] 等效，可避免溢位。最後取整，沒有個別影像的 min-max 延展。

### 1.10 Frequency-domain Combination（頻域組合增強）

將 Homomorphic Filtering（同態濾波）與頻域的 Gaussian High-boost Filtering（高斯高提升濾波）結合。為另一個截止值 Db 建立高斯低通 `H_LP = exp[-D²/(2·Db²)]`。高通 `H_HP = 1 - H_LP`，對應的高提升增益為 `H_HB = 1 + kb·H_HP`。本次 kb > 0。合成頻譜為 `S = H_HB · H_HOMO · Z`。兩濾波器在同一頻譜逐點相乘，只需一次 IFFT，之後套用與方法四相同的裁切及指數回復。

`H_HB` 在 DC 的增益為 1，可在維持低頻處理的同時增加較細尺度的邊緣響應。組合運算作用在對數影像，亮度回復為非線性，因此整體效果有別於直接對原始強度加回細節。高頻加成也會增強顆粒，且高對比處仍可能有亮邊或暗邊。本次用溫和增益保留差異，選用兩濾波器串接。

### 1.11 可重現驗證

小型直接 DFT 以複數加權總和逐 bin 核對一維與二維轉換。測試另涵蓋常數、脈衝、階躍、核對稱、反射索引、偶對稱濾波器、DC、裁切復原、CLI、圖形介面的瀏覽與背景預覽，以及封裝檔案完整性。NumPy 僅用於基本數學和陣列操作。捲積、頻率索引、FFT/IFFT 和所有增益均由本專案實作。

<!-- page -->
## 二、實驗結果｜參數與測量

### 2.1 試驗與選擇

本人為資訊工程系學生，正在學習影像處理，沒有醫學影像、人體解剖或 X 光判讀的專業背景，也較難單憑視覺辨別各方法帶來的改善程度。因此，本次使用 AI 建議第五種方法的技術組合、協助實作程式、分析處理結果，並提出參數調整建議。

每張圖製作 Conservative、Moderate、Strong 及 Selected 四欄原尺寸試驗圖。AI 輔助比較邊緣、細節與顆粒，並參考平均灰階、標準差、平均梯度、Laplacian Variance（拉普拉斯變異數）及飽和像素比例。以 Moderate 為起點，依建議對 2-2.png 降低遮罩增益並調高 γL，對 2-3.png 與 2-5.png 減弱銳化增益，對 2-6.png 調高 γL 並略降組合濾波的高頻增益。選定值用於本次比較，作業範圍限於演算法實作及其影像處理效果。

| 試驗 | α | m / σ | k | γL / γH / D0 | Db / kb |
| --- | --- | --- | --- | --- | --- |
| Conservative | 0.10 | 3 / 0.70 | 1.20 | 0.94 / 1.05 / 0.018 | 0.12 / 0.06 |
| Moderate | 0.18 | 5 / 0.90 | 1.40 | 0.88 / 1.10 / 0.025 | 0.10 / 0.12 |
| Strong | 0.32 | 7 / 1.40 | 1.85 | 0.78 / 1.20 / 0.038 | 0.075 / 0.25 |

### 2.2 選定參數

{{parameters}}

D0 與 Db 以 cycles/pixel 表示，σ 以像素表示。Unsharp Masking（反遮罩）的遮罩增益固定為 1。參數由設定 JSON 傳入管線，表格讀取實際結果清單。

### 2.3 數值指標

平均灰階與母體標準差分別反映整體亮度和灰階分布。平均梯度取對稱邊界下的中央差分 `dx,dy`，對每像素求 `sqrt(dx² + dy²)` 再平均。Laplacian Variance（拉普拉斯變異數）為四鄰域二階差分的母體變異數。總飽和率為 0 和 255 像素占比相加；結果頁同時列出兩端，區分暗端和亮端。較大的梯度或變異數也可能反映顆粒，需搭配圖像判讀。

{{timing}}

後續六張結果頁的圖像觀察由 AI 輔助整理，數值引自結果清單或參數試驗圖。

<!-- page -->
## 二、實驗結果｜2-1.png

{{image:2-1.png}}

High-boost（高提升濾波）讓明暗交界與細粒更顯眼，平均梯度由 4.08 升至 5.87，黑端像素由 0.002% 增為 0.223%。

<!-- page -->
## 二、實驗結果｜2-2.png

{{image:2-2.png}}

High-boost（高提升濾波）讓圖中央的細線及顆粒更顯眼，平均梯度由 3.31 升至 4.80。Combined（頻域組合增強）的平均灰階為 124.70，低於原圖的 127.12，標準差也由 48.53 降至 44.60。

<!-- page -->
## 二、實驗結果｜2-3.png

{{image:2-3.png}}

原圖上方已有大面積暗區，黑端像素占 0.534%。High-boost（高提升濾波）讓高反差交界更明顯，黑端像素增為 0.858%，仍低於 Moderate 試驗圖的 0.934%。

<!-- page -->
## 二、實驗結果｜2-4.png

{{image:2-4.png}}

對照圖的差異較集中於輪廓附近。High-boost（高提升濾波）的平均梯度由 2.66 升至 3.90，Combined（頻域組合增強）則升至 3.04，兩者的黑端像素比例均低於 0.01%。

<!-- page -->
## 二、實驗結果｜2-5.png

{{image:2-5.png}}

明暗交界在銳化後仍然鮮明。High-boost（高提升濾波）的 Selected 試驗圖黑端像素為 0.556%，低於 Moderate 的 0.732%，仍高於原圖的 0.019%。

<!-- page -->
## 二、實驗結果｜2-6.png

{{image:2-6.png}}

原圖平均灰階為 75.86，是六張原圖中最低。Combined（頻域組合增強）的平均灰階為 73.30，高於 Moderate 試驗圖的 71.96。High-boost（高提升濾波）使部分邊緣和背景顆粒更顯眼，平均梯度由 2.84 升至 4.48。

<!-- page -->
## 三、討論與結論

{{discussion}}

Laplacian Operator（拉普拉斯算子）直接作用於灰階快速變化處，實作較簡單、運算成本較低，也容易增強高頻顆粒。Unsharp Masking（反遮罩）可用核尺寸和 σ 調整空間尺度，High-boost Filtering（高提升濾波）則可在同一尺度下增大細節反差。兩方法都可能在強邊緣造成過衝。頻域方法以對數灰階重新分配不同尺度的增益，能壓低大尺度變化，也可能降低原圖的全域對比或平均亮度。

Frequency-domain Combination（頻域組合增強）在同一次逆轉換前結合低頻衰減與高頻補強。頻率響應圖可用來核對增益如何隨頻率變化。六張圖的平均梯度均較 Homomorphic Filtering（同態濾波）高，平均灰階仍低於原圖。

本實驗以五種手動實作的方法完成六張影像的對照。本人較難單靠視覺區分改善程度，因此以 AI 輔助的圖像比較與可重算的指標說明差異。較高梯度可能來自邊緣、細節或噪聲放大，過強的增益也可能造成過度銳化、亮暗暈圈與飽和。本作業的結論限於這六張圖的影像處理結果，不涉及醫學診斷或臨床影像品質評估。

## 四、參考文獻與附錄

1. 課程講義 Chapter 0，第 11 頁（報告格式）；HW #2 Image Sharpening 題目及 e-course2 六張測試圖。
2. R. C. Gonzalez and R. E. Woods, Digital Image Processing, 4th ed., Pearson.
3. J. W. Cooley and J. W. Tukey, “An algorithm for the machine calculation of complex Fourier series,” Mathematics of Computation, 19, 297-301, 1965. DOI 10.1090/S0025-5718-1965-0178586-1.
4. A. V. Oppenheim, R. W. Schafer, and T. G. Stockham, “Nonlinear filtering of multiplied and convolved signals,” Proceedings of the IEEE, 56, 1264-1291, 1968. DOI 10.1109/PROC.1968.6570.
5. 前次作業 Image Processing Spatial Image Enhancement，commit 6a15a6b。此次沿用 Python src-layout、uv、Tkinter 導覽與背景預覽、Markdown 至 ReportLab PDF、PyInstaller 規格檔及本機 ZIP 驗證等架構模式。銳化演算法的開發過程使用第 2.1 節所述的 AI 輔助。

### 原始程式碼（Source Code）

https://github.com/whoami13579/Image_Processing_Image_Sharpening

儲存庫包含原始程式碼、五種演算法的實作、設定檔、測試與報告生成工具。

### 附錄｜重現入口

```
uv sync --locked --python 3.12
uv run python -m pytest -q
uv run python -m image_sharpening --input HW2_test_image --output outputs
uv run python scripts/compare_parameters.py
uv run python report/build_report.py
uv run python -m PyInstaller --noconfirm --clean ImageSharpening.spec
uv run python scripts/build_submission.py
```
