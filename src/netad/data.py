"""NSL-KDD loader + a schema-compatible synthetic generator.

Real data: put KDDTrain+.txt and KDDTest+.txt (NSL-KDD) in ./data/.
Without them, `load_data` returns a synthetic dataset with the SAME 41
features, in which the test split contains an attack family (R2L) that never
appears in training. That mirrors the real research problem: detecting
attacks you have never seen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["protocol_type", "service", "flag"]
FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
# heavy-tailed count/byte columns get log1p before scaling
LOG_COLS = ["duration", "src_bytes", "dst_bytes", "hot", "num_compromised", "num_root",
            "num_file_creations", "num_access_files", "count", "srv_count",
            "dst_host_count", "dst_host_srv_count"]
RAW_COLUMNS = FEATURES + ["label", "difficulty"]

ATTACK_FAMILY = {
    **dict.fromkeys(["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
                     "udpstorm", "processtable", "worm", "mailbomb"], "dos"),
    **dict.fromkeys(["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"], "probe"),
    **dict.fromkeys(["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
                     "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
                     "httptunnel", "sendmail", "named"], "r2l"),
    **dict.fromkeys(["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
                     "xterm", "ps"], "u2r"),
    "normal": "normal",
}


# ----------------------------------------------------------------------------- real data
def _read_nsl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    df["family"] = df["label"].map(ATTACK_FAMILY).fillna("other_attack")
    df["is_attack"] = (df["label"] != "normal").astype(int)
    return df.drop(columns=["difficulty"])


# ------------------------------------------------------------------------- synthetic data
def _blank(n: int) -> dict[str, np.ndarray]:
    d = {c: np.zeros(n) for c in NUMERIC}
    d.update({c: np.array([""] * n, dtype=object) for c in CATEGORICAL})
    return d


def _synth_family(family: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    d = _blank(n)
    pick = lambda opts, p: rng.choice(opts, size=n, p=p)  # noqa: E731
    beta = lambda a, b: rng.beta(a, b, size=n)  # noqa: E731

    if family == "normal":
        d["protocol_type"] = pick(["tcp", "udp", "icmp"], [0.80, 0.15, 0.05])
        d["service"] = pick(["http", "smtp", "ftp_data", "domain_u", "private", "ssh", "other"],
                            [0.45, 0.12, 0.08, 0.12, 0.05, 0.05, 0.13])
        d["flag"] = pick(["SF", "S0", "REJ", "RSTO"], [0.92, 0.03, 0.03, 0.02])
        d["duration"] = rng.exponential(15, n) * (rng.random(n) < 0.3)
        d["src_bytes"] = rng.lognormal(6.0, 1.5, n)
        d["dst_bytes"] = rng.lognormal(7.0, 1.8, n)
        d["logged_in"] = (rng.random(n) < 0.7).astype(float)
        d["count"] = np.clip(rng.poisson(8, n), 1, 511)
        d["srv_count"] = np.clip(rng.poisson(8, n), 1, 511)
        d["serror_rate"], d["srv_serror_rate"] = beta(1, 40), beta(1, 40)
        d["rerror_rate"], d["srv_rerror_rate"] = beta(1, 40), beta(1, 40)
        d["same_srv_rate"], d["diff_srv_rate"] = beta(9, 1), beta(1, 12)
        d["srv_diff_host_rate"] = beta(1, 8)
        d["dst_host_count"] = rng.integers(60, 256, n)
        d["dst_host_srv_count"] = rng.integers(40, 256, n)
        d["dst_host_same_srv_rate"], d["dst_host_diff_srv_rate"] = beta(8, 2), beta(1, 15)
        d["dst_host_same_src_port_rate"], d["dst_host_srv_diff_host_rate"] = beta(1, 6), beta(1, 12)
        d["dst_host_serror_rate"], d["dst_host_srv_serror_rate"] = beta(1, 40), beta(1, 40)
        d["dst_host_rerror_rate"], d["dst_host_srv_rerror_rate"] = beta(1, 40), beta(1, 40)
        d["hot"] = rng.poisson(0.05, n)
    elif family == "dos":      # SYN-flood style: huge counts, SYN errors, no payload
        d["protocol_type"] = pick(["tcp", "icmp", "udp"], [0.7, 0.25, 0.05])
        d["service"] = pick(["private", "http", "ecr_i", "other"], [0.4, 0.3, 0.2, 0.1])
        d["flag"] = pick(["S0", "REJ", "SF"], [0.8, 0.15, 0.05])
        d["src_bytes"] = rng.lognormal(1.0, 2.0, n) * (rng.random(n) < 0.3)
        d["count"] = np.clip(rng.poisson(250, n), 1, 511)
        d["srv_count"] = np.clip(rng.poisson(240, n), 1, 511)
        d["serror_rate"], d["srv_serror_rate"] = beta(12, 1), beta(12, 1)
        d["same_srv_rate"], d["diff_srv_rate"] = beta(30, 1), beta(1, 40)
        d["dst_host_count"] = rng.integers(200, 256, n)
        d["dst_host_srv_count"] = rng.integers(5, 40, n)
        d["dst_host_same_srv_rate"], d["dst_host_serror_rate"] = beta(2, 6), beta(12, 1)
        d["dst_host_srv_serror_rate"] = beta(12, 1)
    elif family == "probe":    # port scans: many services, reject errors
        d["protocol_type"] = pick(["tcp", "icmp", "udp"], [0.75, 0.2, 0.05])
        d["service"] = pick(["private", "other", "http", "ftp_data", "ssh"], [0.5, 0.2, 0.1, 0.1, 0.1])
        d["flag"] = pick(["REJ", "RSTO", "SF", "S0"], [0.5, 0.2, 0.2, 0.1])
        d["src_bytes"] = rng.lognormal(2.0, 1.5, n) * (rng.random(n) < 0.4)
        d["count"] = np.clip(rng.poisson(35, n), 1, 511)
        d["srv_count"] = np.clip(rng.poisson(4, n), 1, 511)
        d["rerror_rate"], d["srv_rerror_rate"] = beta(6, 3), beta(6, 3)
        d["same_srv_rate"], d["diff_srv_rate"] = beta(2, 6), beta(6, 3)
        d["dst_host_count"] = rng.integers(100, 256, n)
        d["dst_host_srv_count"] = rng.integers(1, 20, n)
        d["dst_host_same_srv_rate"], d["dst_host_diff_srv_rate"] = beta(1, 8), beta(6, 3)
        d["dst_host_rerror_rate"], d["dst_host_srv_rerror_rate"] = beta(6, 3), beta(6, 3)
    elif family == "r2l":      # remote-to-local: low-and-slow, close to normal traffic
        d["protocol_type"] = pick(["tcp", "udp"], [0.95, 0.05])
        d["service"] = pick(["ftp", "telnet", "http", "smtp", "ssh"], [0.3, 0.3, 0.2, 0.1, 0.1])
        d["flag"] = pick(["SF", "RSTO"], [0.92, 0.08])
        d["duration"] = rng.exponential(60, n)
        d["src_bytes"] = rng.lognormal(6.5, 1.2, n)
        d["dst_bytes"] = rng.lognormal(7.2, 1.5, n)
        d["logged_in"] = (rng.random(n) < 0.55).astype(float)
        d["hot"] = rng.poisson(2.5, n)
        d["num_failed_logins"] = rng.poisson(0.9, n)
        d["is_guest_login"] = (rng.random(n) < 0.35).astype(float)
        d["count"] = np.clip(rng.poisson(4, n), 1, 511)
        d["srv_count"] = np.clip(rng.poisson(4, n), 1, 511)
        d["serror_rate"], d["rerror_rate"] = beta(1, 30), beta(1, 20)
        d["same_srv_rate"], d["diff_srv_rate"] = beta(9, 1), beta(1, 12)
        d["dst_host_count"] = rng.integers(1, 120, n)
        d["dst_host_srv_count"] = rng.integers(1, 80, n)
        d["dst_host_same_srv_rate"] = beta(8, 2)
        d["dst_host_same_src_port_rate"] = beta(1, 6)
    else:
        raise ValueError(f"unknown family {family}")
    out = pd.DataFrame(d)[FEATURES]
    out["label"], out["family"] = family, family
    out["is_attack"] = int(family != "normal")
    return out


def make_synthetic(seed: int = 42, scale: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train has normal+dos+probe. Test additionally has the NOVEL r2l family."""
    rng = np.random.default_rng(seed)
    n = lambda k: max(int(k * scale), 20)  # noqa: E731
    train = pd.concat([_synth_family("normal", n(20000), rng), _synth_family("dos", n(2000), rng),
                       _synth_family("probe", n(1500), rng)])
    test = pd.concat([_synth_family("normal", n(5000), rng), _synth_family("dos", n(1500), rng),
                      _synth_family("probe", n(1000), rng), _synth_family("r2l", n(800), rng)])
    shuffle = lambda df: df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # noqa: E731
    return shuffle(train), shuffle(test)


def load_data(data_dir: str | Path = "data", seed: int = 42,
              scale: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    data_dir = Path(data_dir)
    tr, te = data_dir / "KDDTrain+.txt", data_dir / "KDDTest+.txt"
    if tr.exists() and te.exists():
        return _read_nsl(tr), _read_nsl(te), "NSL-KDD"
    train, test = make_synthetic(seed, scale)
    return train, test, "synthetic"


# ------------------------------------------------------------------------ preprocessing
def build_preprocessor() -> ColumnTransformer:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    log_cols = [c for c in LOG_COLS if c in NUMERIC]
    other_num = [c for c in NUMERIC if c not in log_cols]
    log_pipe = Pipeline([("log", FunctionTransformer(np.log1p, validate=False)),
                         ("scale", StandardScaler())])
    return ColumnTransformer([
        ("log", log_pipe, log_cols),
        ("num", StandardScaler(), other_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
