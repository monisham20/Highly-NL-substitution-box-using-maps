import numpy as np

# ================================================================
#                    8-BIT LFSR FUNCTION
# ================================================================
def lfsr_8bit(seed):
    """8-bit LFSR: x^8 + x^6 + x^5 + x^4 + 1"""
    reg = seed & 0xFF
    while True:
        bit = ((reg >> 7) ^ (reg >> 5) ^ (reg >> 4) ^ (reg >> 3)) & 1
        reg = ((reg << 1) & 0xFF) | bit
        yield reg & 0xFF


# --- Generate LFSR S-Box ---
def generate_lfsr_sbox(seed=0xA5):
    """
    Produce 255 unique values from LFSR (skipping duplicates) then append 0
    to create a 256-element S-box (as in earlier code).
    """
    gen = lfsr_8bit(seed)
    sbox = []
    used = set()
    while len(sbox) < 255:
        v = next(gen)
        if v not in used:
            used.add(v)
            sbox.append(v)
    # append 0 to complete 256 entries
    if 0 not in used:
        sbox.append(0)
    else:
        # if 0 already present (unlikely), fill with any missing values
        for x in range(256):
            if x not in used:
                sbox.append(x)
                used.add(x)
            if len(sbox) == 256:
                break
    return np.array(sbox, dtype=np.uint8)


# ================================================================
#                CHAOTIC LOGISTIC MAP S-BOX
# ================================================================
def generate_dynamic_sbox(base_sbox, mu=3.99, x0=0.31, iterations=500):
    """
    Use logistic map to produce a single-byte xor_val, then XOR with base_sbox.
    Returns: chaotic_sbox (np.uint8 array), xor_val, mu, x0, iterations
    """
    x = x0
    for _ in range(iterations):
        x = mu * x * (1 - x)
    xor_val = int((abs(x) * 10000) % 256)
    xor_byte = np.uint8(xor_val)
    chaotic = np.array([int(b) ^ int(xor_byte) for b in base_sbox], dtype=np.uint8)
    return chaotic, xor_val, mu, x0, iterations


# ================================================================
#                   AES S-BOX (INPUT FOR CHAOS)
# ================================================================
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


# ================================================================
#         MERGE RULE: APPLY FOR ALL INDICES (with collision handling)
# ================================================================
def merge_sboxes_full(lfsr_sbox, chaotic_sbox):
    """
    Full 0..255 mapping:
      For each i: attempt NEW[ chaotic_sbox[i] ] = lfsr_sbox[i]
    If some positions collide (already filled), we leave them empty and
    fill later with remaining unused values to preserve bijection.
    """
    NEW = [-1] * 256
    used_values = set()
    collisions = []

    # First pass: place values where positions are free
    for i in range(256):
        pos = int(chaotic_sbox[i])
        val = int(lfsr_sbox[i])
        if NEW[pos] == -1:
            NEW[pos] = val
            used_values.add(val)
        else:
            collisions.append((i, pos, val))

    # Fill remaining empty positions with any values not used yet
    remaining_values = [v for v in range(256) if v not in used_values]
    r_idx = 0
    for idx in range(256):
        if NEW[idx] == -1:
            NEW[idx] = remaining_values[r_idx]
            r_idx += 1

    return np.array(NEW, dtype=np.uint8), collisions


# ================================================================
#                      METRIC FUNCTIONS
# ================================================================
def hamming_weight(x):
    return bin(x).count("1")


def calculate_nonlinearity(boolean_function):
    # boolean_function: list/array of 256 bits (0/1)
    walsh = np.array([
        sum((-1) ** (int(boolean_function[x]) ^ (bin(k & x).count("1") % 2))
            for x in range(256))
        for k in range(256)
    ])
    return int(128 - (np.max(np.abs(walsh)) / 2))


def calculate_nl_function(sbox):
    max_corr = 0
    for a in range(1, 256):
        for b in range(1, 256):
            corr = 0
            for x in range(256):
                parity = (bin(x & a).count("1") + bin(int(sbox[x]) & b).count("1")) % 2
                corr += (-1) ** parity
            max_corr = max(max_corr, abs(corr))
    return int(128 - max_corr / 2)


def calculate_sac(sbox):
    total = 0
    for i in range(8):
        flips = [int(sbox[x]) ^ int(sbox[x ^ (1 << i)]) for x in range(256)]
        total += sum(hamming_weight(f) for f in flips)
    return total / (256 * 8 * 8)


def calculate_bic_nl(sbox):
    nl_values = []
    for j in range(8):
        bits = [((int(sbox[x]) >> j) & 1) for x in range(256)]
        nl_values.append(calculate_nonlinearity(bits))
    return int(np.mean(nl_values)), min(nl_values), max(nl_values)


def calculate_bic_sac(sbox):
    flips_total, count = 0, 0
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            flip = sum(
                (((int(sbox[x]) >> j) & 1) != ((int(sbox[x ^ (1 << i)]) >> j) & 1))
                for x in range(256)
            )
            flips_total += flip / 256.0
            count += 1
    return (flips_total / count) if count else 0.0


def calculate_lap(sbox):
    max_lap = 0.0
    for a in range(1, 256):
        for b in range(1, 256):
            cnt = sum(
                1 for x in range(256)
                if hamming_weight((x & a) ^ (int(sbox[x]) & b)) % 2 == 0
            )
            lap = abs(cnt - 128) / 256.0
            max_lap = max(max_lap, lap)
    return max_lap


def calculate_dap(sbox):
    max_dap = 0.0
    for dx in range(1, 256):
        diff = [int(sbox[x]) ^ int(sbox[x ^ dx]) for x in range(256)]
        counts = np.bincount(diff, minlength=256)
        max_dap = max(max_dap, np.max(counts) / 256.0)
    return max_dap


def check_bijectivity(sbox):
    return len(set(int(x) for x in sbox)) == 256


def evaluate_sbox(sbox, name="S-box"):
    print(f"\n=========== {name} ===========")
    print("Bijective:", check_bijectivity(sbox))
    print("Nonlinearity:", calculate_nl_function(sbox))
    avg_nl, mn, mx = calculate_bic_nl(sbox)
    print("BIC NL (avg/min/max):", avg_nl, mn, mx)
    print("SAC:", round(calculate_sac(sbox), 6))
    print("BIC SAC:", round(calculate_bic_sac(sbox), 6))
    print("LAP:", round(calculate_lap(sbox), 6))
    print("DAP:", round(calculate_dap(sbox), 6))
    print("=================================\n")


# ================================================================
#                         PRINT HELPERS
# ================================================================
def print_sbox_16x16(sbox, title="S-box"):
    print(f"\n=== {title} (16x16 HEX) ===")
    mat = sbox.reshape(16, 16)
    for row in mat:
        print(" ".join(f"{v:02X}" for v in row))


# ================================================================
#                              MAIN
# ================================================================
if __name__ == "__main__":
    # Parameters (change as desired)
    LFSR_SEED = 0xA5
    MU = 3.99
    X0 = 0.31
    ITER = 500

    # Generate S-boxes
    lfsr_sbox = generate_lfsr_sbox(seed=LFSR_SEED)
    chaotic_sbox, xor_val, mu_used, x0_used, iters_used = generate_dynamic_sbox(
        AES_SBOX, mu=MU, x0=X0, iterations=ITER
    )

    # Merge full 0..255 mapping
    combined_sbox, collisions = merge_sboxes_full(lfsr_sbox, chaotic_sbox)

    # Print parameters and S-boxes
    print("\n=== PARAMETERS ===")
    print(f"LFSR seed: 0x{LFSR_SEED:02X}")
    print(f"Logistic mu: {mu_used}, x0: {x0_used}, iterations: {iters_used}")
    print(f"Computed chaos xor_val: {xor_val} (0x{xor_val:02X})")

    print_sbox_16x16(lfsr_sbox, "LFSR S-BOX")
    print_sbox_16x16(chaotic_sbox, "CHAOTIC LOGISTIC MAP S-BOX")
    print_sbox_16x16(combined_sbox, "FINAL COMBINED S-BOX (chaotic_pos <- lfsr_val)")

    if collisions:
        print(f"\nNote: {len(collisions)} collisions occurred during mapping (they were resolved by filling remaining slots).")

    # Evaluate metrics
    evaluate_sbox(lfsr_sbox, "LFSR S-Box")
    evaluate_sbox(chaotic_sbox, "Chaotic Logistic S-Box")
    evaluate_sbox(combined_sbox, "Final Combined S-Box")
