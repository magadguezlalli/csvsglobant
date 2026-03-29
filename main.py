import pandas as pd
import glob
import os
import ast
import re

# ── CARGAR Y UNIR ──────────────────────────────────────────────────────────────
folder = "./datasets"
files = glob.glob(os.path.join(folder, "*.csv"))

dfs = [pd.read_csv(file) for file in files]
df = pd.concat(dfs, ignore_index=True)
print(f"Filas originales: {len(df)}")

# ── LIMPIEZA BÁSICA ────────────────────────────────────────────────────────────
df.drop_duplicates(inplace=True)
df.dropna(subset=["total_sizes"], inplace=True)
print(f"Tras limpieza: {len(df)} filas")

# parse datetime
df["scrapping_datetime"] = pd.to_datetime(df["scrapping_datetime"], format="%d-%m-%y %H:%M")

# rename columns
df.rename(columns={"retailer": "vendor", "brand_name": "brand", "product_category": "category",
                   "product_name": "product", "mrp": "MRP"}, inplace=True)

# remove columns
df.drop(["pdp_url", "description"], axis=1, inplace=True)

# ── FUNCIONES ──────────────────────────────────────────────────────────────────

def parse_sizes(size_field):
    if pd.isna(size_field):
        return []
    s = str(size_field).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]


def is_bra_size(size):
    size = str(size).strip().upper()
    if re.fullmatch(r"(X{0,3}S|X{0,3}M|X{0,3}L|S|M|L)", size):
        return False
    if re.fullmatch(r"\d{1,2}", size):
        return False
    if re.fullmatch(r"[A-Z]{1,2}", size):
        return False
    if re.fullmatch(r"\d+X", size):
        return True
    if re.match(r"^\d{2}[A-Z]", size):
        return True
    return False


def get_size_group(size):
    size = str(size).strip().upper()
    if re.fullmatch(r"\d+X", size):
        return "Extra Large"
    match = re.match(r"^(\d{2})", size)
    if match:
        underbust = int(match.group(1))
        if underbust in [30, 32]:       return "Small"
        elif underbust in [34, 36]:     return "Medium"
        elif underbust in [38, 40]:     return "Large"
        elif underbust in [42, 44, 46]: return "Extra Large"
    return "Unknown"


def classify_product(row):
    sizes = parse_sizes(row.get("total_sizes", ""))
    bra_sizes = [s for s in sizes if is_bra_size(s)]

    if not bra_sizes:
        return pd.Series({"is_bra": False, "size_group": None,
                          "bra_offered_count": 0, "size": None})

    groups = [get_size_group(s) for s in bra_sizes]
    dominant_group = pd.Series(groups).value_counts().idxmax()

    # Primer size que pertenece al grupo dominante
    dominant_size = next(s for s, g in zip(bra_sizes, groups) if g == dominant_group)

    return pd.Series({"is_bra": True, "size_group": dominant_group,
                      "bra_offered_count": len(bra_sizes), "size": dominant_size})


def count_available_bras(row):
    available = parse_sizes(row.get("available_size", ""))
    return len([s for s in available if is_bra_size(s)])


# ── CLASIFICAR ─────────────────────────────────────────────────────────────────
print("Clasificando productos...")
df[["is_bra", "size_group", "bra_offered_count", "size"]] = df.apply(classify_product, axis=1)

# ── IS_AVAILABLE ───────────────────────────────────────────────────────────────
df["available_count"] = df.apply(count_available_bras, axis=1)

df["is_available"] = df.apply(
    lambda r: ("Yes" if r["available_count"] > 0 else "No") if r["is_bra"] else None, axis=1
)

# ── AVAILABILITY_RATIO Y STATUS ────────────────────────────────────────────────
xl = df[(df["is_bra"]) & (df["size_group"] == "Extra Large")].groupby("vendor").agg(
    xl_offered=("bra_offered_count", "sum"),
    xl_available=("available_count", "sum")
).reset_index()

xl["availability_ratio"] = (xl["xl_available"] / xl["xl_offered"]).clip(upper=1.0).round(4)

def get_status(pct):
    if pct < 0.30:   return "Sanction"
    elif pct < 0.50: return "Warning"
    else:            return "OK"

xl["status"] = xl["availability_ratio"].apply(get_status)

# Merge de vuelta al dataset completo
df = df.merge(xl[["vendor", "availability_ratio", "status"]], on="vendor", how="left")

# Solo Extra Large bras tienen status y availability_ratio, el resto None
df["availability_ratio"] = df.apply(
    lambda r: r["availability_ratio"] if (r["is_bra"] and r["size_group"] == "Extra Large") else None, axis=1
)
df["status"] = df.apply(
    lambda r: r["status"] if (r["is_bra"] and r["size_group"] == "Extra Large") else None, axis=1
)


df.to_csv("dataset_complete.csv", index=False)
print(f"\nExportado: dataset_complete.csv — {len(df)} filas, {len(df.columns)} columnas")
print(f"   Bras: {df['is_bra'].sum()} | No bras: {(~df['is_bra']).sum()}")

print("\nStatus por vendedor (Extra Large):")
print(xl[["vendor", "availability_ratio", "status"]].to_string(index=False))
