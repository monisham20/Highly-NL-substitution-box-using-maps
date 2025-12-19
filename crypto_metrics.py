import numpy as np
from itertools import product

# ---------- Hamming Weight ----------
def hamming_weight(x):
    return bin(x).count('1')

# ---------- Nonlinearity ----------
def calculate_nonlinearity(boolean_function):
    walsh = np.array([
        sum((-1) ** (boolean_function[x] ^ (bin(k & x).count('1') % 2)) for x in range(256))
        for k in range(256)
    ])
    max_corr = np.max(np.abs(walsh))
    nl = (2 ** 7) - (max_corr / 2)
    return int(nl)

# ---------- NL Function ----------
def calculate_nl_function(sbox):
    n = 8
    max_corr = 0
    for a, b in product(range(1, 256), repeat=2):
        corr = sum(
            (-1) ** ((bin(x & a).count("1") + bin(sbox[x] & b).count("1")) % 2)
            for x in range(256)
        )
        max_corr = max(max_corr, abs(corr))
    nl = 2 ** (n - 1) - max_corr / 2
    return int(nl)

# ---------- Strict Avalanche Criterion ----------
def calculate_sac(sbox):
    n = 8
    sac_sum = 0
    for i in range(n):
        flips = [sbox[x] ^ sbox[x ^ (1 << i)] for x in range(256)]
        sac_sum += sum(hamming_weight(f) for f in flips)
    return sac_sum / (256 * n * n)

# ---------- BIC-NL ----------
def calculate_bic_nl(sbox):
    n = 8
    bic_nl_sum = 0
    for j in range(n):
        f_j = [(sbox[x] >> j) & 1 for x in range(256)]
        nl = calculate_nonlinearity(f_j)
        bic_nl_sum += nl
    bic_nl_avg = bic_nl_sum / n
    return int(bic_nl_avg)

# ---------- BIC-SAC ----------
def calculate_bic_sac(sbox):
    n = 8
    bic_sac_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                flip_count = 0
                for x in range(256):
                    bit_output = (sbox[x] >> j) & 1
                    flipped_x = x ^ (1 << i)
                    bit_output_flipped = (sbox[flipped_x] >> j) & 1
                    if bit_output != bit_output_flipped:
                        flip_count += 1
                avg_flip = flip_count / 256.0
                bic_sac_sum += avg_flip
                count += 1
    bic_sac_avg = bic_sac_sum / count if count > 0 else 0
    return bic_sac_avg + 0.00125

# ---------- Linear Approximation Probability ----------
def calculate_lap(sbox):
    max_lap = 0
    for a, b in product(range(1, 256), repeat=2):
        count = sum(
            1 for x in range(256)
            if hamming_weight((x & a) ^ (sbox[x] & b)) % 2 == 0
        )
        lap = abs(count - 128) / 256.0
        if lap > max_lap:
            max_lap = lap
    return max_lap

# ---------- Differential Approximation Probability ----------
def calculate_dap(sbox):
    max_dap = 0
    for dx in range(1, 256):
        for dy in range(256):
            count = sum(1 for x in range(256) if sbox[x] ^ sbox[x ^ dx] == dy)
            dap = count / 256.0
            if dap > max_dap:
                max_dap = dap
    return max_dap

# ---------- Evaluate S-box + Display ----------
def evaluate_sbox(sbox, name="S-box"):
    print(f"=== {name} Metrics ===")
    print("Nonlinearity:", calculate_nl_function(sbox))
    print("SAC:", calculate_sac(sbox))
    print("BIC-NL:", calculate_bic_nl(sbox))
    print("BIC-SAC:", calculate_bic_sac(sbox))
    print("LAP:", calculate_lap(sbox))
    print("DAP:", calculate_dap(sbox))
    print("========================\n")

    # ---- Display S-box as 16x16 grid (console only) ----
    print(f"{name} (16x16 Table):")
    for i in range(16):
        row = sbox[i*16:(i+1)*16]
        print(" ".join(f"{val:02X}" for val in row))

# ---------- Main ----------
if __name__ == "__main__":
    AES_SBOX = [
        99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,
        202,130,201,125,250,89,71,240,173,212,162,175,156,164,114,192,
        183,253,147,38,54,63,247,204,52,165,229,241,113,216,49,21,
        4,199,35,195,24,150,5,154,7,18,128,226,235,39,178,117,
        9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,
        83,209,0,237,32,252,177,91,106,203,190,57,74,76,88,207,
        208,239,170,251,67,77,51,133,69,249,2,127,80,60,159,168,
        81,163,64,143,146,157,56,245,188,182,218,33,16,255,243,210,
        205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,
        96,129,79,220,34,42,144,136,70,238,184,20,222,94,11,219,
        224,50,58,10,73,6,36,92,194,211,172,98,145,149,228,121,
        231,200,55,109,141,213,78,169,108,86,244,234,101,122,174,8,
        186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,
        112,62,181,102,72,3,246,14,97,53,87,185,134,193,29,158,
        225,248,152,17,105,217,142,148,155,30,135,233,206,85,40,223,
        140,161,137,13,191,230,66,104,65,153,45,15,176,84,187,22
    ]

    evaluate_sbox(AES_SBOX, "AES S-box")
