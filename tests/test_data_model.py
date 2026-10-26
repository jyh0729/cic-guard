"""데이터 정리·학습 스크립트 테스트 (가짜 CIC 형식 데이터)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from data import load_clean  # noqa: E402
from features import CSV_FEATURES  # noqa: E402
from make_sample import make  # noqa: E402


def test_cleaning_removes_header_inf_and_duplicates(tmp_path):
    csv = tmp_path / "s.csv"
    make().to_csv(csv, index=False)
    df, rep = load_clean([str(csv)])
    assert rep["after_header_rows"] == rep["raw_rows"] - 1
    assert rep["rows_with_missing_or_inf"] >= 30
    assert rep["after_dedup"] < rep["after_dropna"]
    assert not df[CSV_FEATURES].isna().any().any()
    assert not df.duplicated(subset=CSV_FEATURES + ["Label"]).any()
    assert set(df["Label"]) == {"Benign", "FTP-BruteForce", "SSH-Bruteforce"}


def test_training_writes_model_and_metrics(tmp_path):
    csv = tmp_path / "s.csv"
    make().to_csv(csv, index=False)
    subprocess.run([sys.executable, str(ROOT / "model.py"), "--csv", str(csv), "--out-dir", str(tmp_path)],
                   check=True, cwd=ROOT, capture_output=True)
    m = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert (tmp_path / "model.joblib").exists()
    assert {"baselines", "test", "unseen_attack", "warnings"} <= set(m)
    assert "port_rule_21_22" in m["baselines"]
    assert m["split"]["train"] > m["split"]["test"]
