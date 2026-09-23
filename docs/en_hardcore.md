# HardCore (MemriNeurons/cores.py)

**HardCore** works in conjunction with **MemriBoard**. It includes a set of methods applicable to a memristor crossbar array connected to a computer via a board. When creating an instance of the class, you must specify one argument:

- `conn` – connection to the memristor board (real hardware or simulator).

## Methods

### `set_wr(self)`
Switch to write-read mode. The weight is calculated using the formula for the read/write loop:  
`Weight = R_load / (R_load + R_mem)`, where `R_load = 3000 Ω`.

### `set_mvm(self)`
Switch to matrix-vector multiplication mode. The weight is calculated using the formula for the matrix multiplication loop:  
`Weight = R_tia / R_mem`, where `R_tia = 330 Ω`.

### `read_one_weight(self, bl, wl)`
Read the weight value of a single cell at coordinates `(bl, wl)`.  
**Returns:** The weight value according to the currently selected mode.

### `read_raw_weights(self)`
Read the raw weight values of all cells in the memristor crossbar array.  
**Returns:** A weight matrix.

### `write_weight(self, bl, wl, weight_value, silent=True)`
Write a single weight to the cell at coordinates `(bl, wl)` using the **write & verify** method.

> **⚠️ CAUTION:** The write pulse settings are class attributes of `HardCore`. If you wish to change them, do so carefully – **do not apply more than 1.5 V** to real memristors. If you are working with a simulator, you can apply higher voltages (e.g., 3 V).

**Returns:** The written weight value and weight history.

### `write_matrix(self, matrix, silent=True)`
Write a weight matrix. Make sure the matrix size does not exceed the crossbar array dimensions (smaller is acceptable).  
**Returns:** Two lists – weight values and resistance values.

### `multiplication(self, x, bl, wl, scale_x=1, scale_w=1, sign_w=1)`
Perform analog multiplication of two numbers via the read/write loop. The input value `x` is multiplied by the weight `w` stored in the cell at coordinates `(bl, wl)`.

**Parameters:**
- `scale_x` – scaling factor for the input
- `scale_w` – scaling factor for the weight
- `sign_w` – sign multiplier for the weight

- `sign_w` – sign multiplier for the weight

**Mathematical formulation:**

The voltage applied to the memristor is calculated as:

```
V_bl = x / scale_x
```

The multiplication formula is:

```
M = V_bl · K_x · W_mem · K_w
```

Where:
- `X = V_bl · K_x` — the input value
- `W = W_mem · K_w` — the weight value
- `M = X · W` — the target multiplication result

From this, we derive:
- `V_bl = X / K_x`
- `W_mem = W / K_w`

The board returns:
```
M_b = V_bl · W_mem
```

Therefore, the actual multiplication result is:
```
M = M_b · K_x · K_w
```

> **⚠️ IMPORTANT:** `V_bl` must be **≤ 300 mV** – this is the maximum voltage that does not alter the memristor's resistance. If the calculated value exceeds 300 mV, the protection system will clamp it to 300 mV, and a warning will be printed.

**Returns:** The multiplication result.

### `dot_product(self, x, wl, scale_x=1, scale_w=1)`
Perform analog dot product of two vectors via the matrix-vector multiplication loop. The input vector `x` is multiplied by the weight vector stored in column `wl` of the crossbar array.

**Operation:**
- Multiplication follows **Ohm's law**
- Summation follows **Kirchhoff's current law**

**Built-in safety protections:**
1. **Voltage limit** – no input exceeds 300 mV
2. **Current limit** – total current on column `wl` does not exceed **15 mA**

**Returns:** The dot product result.
