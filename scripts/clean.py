import re
import sys
from pathlib import Path

# tqdm bar: "Training beta:   0%|          | 0/522 [00:00<?, ?it/s]"
# or block chars: "Training beta: 100%|██████████| 522/522 [02:12<00:00,  3.95it/s]"
TRAINING_BETA_TQDM = re.compile(
    r"^Training beta:\s+\d+%\|[^|]*\|\s*\d+/\d+\s+\[[^\]]*\]\s*$"
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO / "outputs" / "beta_training.log"


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    out_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else in_path.with_name(f"{in_path.stem}_clean{in_path.suffix}")
    )

    kept = 0
    dropped = 0
    lines_out = []
    with in_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            body = line.rstrip("\r\n")
            if TRAINING_BETA_TQDM.match(body):
                dropped += 1
                continue
            lines_out.append(line)
            kept += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(lines_out)

    print(f"done: kept {kept} lines, dropped {dropped} tqdm lines -> {out_path}")


if __name__ == "__main__":
    main()
