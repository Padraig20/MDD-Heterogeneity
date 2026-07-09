#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import re



# individuals here go into the test split; ~20% + 5 individuals from MLP average
INDIVIDUAL_IDS = [
    "1001", "1002", "1003", "1004", "1005",
    "3", "7", "8", "11", "19", "24", "37", "45", "56", "60",
    "65", "82", "84", "99", "107", "112", "117", "119", "123", "127",
    "128", "137", "139", "141", "152", "168", "169", "172", "182", "185",
    "187", "189", "191", "192", "223", "232", "244", "246", "247", "253",
    "256", "257", "260", "265", "267", "272", "284", "287", "291", "302",
    "303", "304", "306", "307", "309", "313", "316", "328", "332", "355",
    "356", "365", "377", "384", "394", "399", "404", "410", "417", "418",
    "419", "420", "423", "434", "437", "439", "458", "466", "472", "485",
    "490", "494", "495", "499", "503", "529", "534", "540", "543", "562",
    "567", "572", "579", "585", "586", "590", "593", "599", "616", "617",
    "621", "622", "628", "635", "637", "641", "652", "661", "668", "671",
    "672", "673", "675", "680", "682", "688", "693", "701", "708", "712",
    "716", "717", "719", "721", "731", "735", "738", "751", "753", "756",
    "763", "765", "772", "777", "780", "782", "784", "789", "794", "798",
    "805", "806", "810", "813", "831", "832", "847", "848", "849", "862",
    "870", "874", "881", "886", "893", "900", "905", "906", "907", "908",
    "924", "927", "931", "933", "938", "948", "956", "957", "958", "960",
    "970", "975", "979", "982", "986", "987", "990", "993", "995", "997",
    "1007", "1012", "1027", "1029", "1031", "1033", "1045", "1047", "1049",
    "1057", "1064", "1066", "1071", "1073", "1080"
    ]


# Metadata columns that should be kept in BOTH train and test files.
META_COLUMNS = {"gene", "chrom", "tss"}


def normalize_id(x: str) -> str:
    """
    Extract individual ID from a column name.

    Examples:
        "1001"         -> "1001"
        "OneK1K_1001" -> "1001"
    """
    x = x.strip()

    if x in INDIVIDUAL_IDS:
        return x

    # Match trailing numeric ID, e.g. OneK1K_1001 -> 1001
    m = re.search(r"(\d+)$", x)
    if m:
        return m.group(1)

    return x


def split_file(input_file: Path, train_dir: Path, test_dir: Path, test_ids: set[str]) -> None:
    train_file = train_dir / input_file.name
    test_file = test_dir / input_file.name

    with input_file.open("r", newline="") as fin:
        reader = csv.reader(fin)

        try:
            header = next(reader)
        except StopIteration:
            print(f"[SKIP] Empty file: {input_file}")
            return

        meta_idx = []
        train_idx = []
        test_idx = []

        for i, col in enumerate(header):
            col_clean = col.strip()

            if col_clean in META_COLUMNS:
                meta_idx.append(i)
            else:
                indiv_id = normalize_id(col_clean)
                if indiv_id in test_ids:
                    test_idx.append(i)
                else:
                    train_idx.append(i)

        train_keep_idx = meta_idx + train_idx
        test_keep_idx = meta_idx + test_idx

        missing_test_ids = sorted(
            test_ids
            - {normalize_id(header[i].strip()) for i in test_idx}
        )

        if missing_test_ids:
            print(
                f"[WARN] {input_file.name}: "
                f"{len(missing_test_ids)} test IDs not found in this file: {', '.join(missing_test_ids)}"
            )

        with train_file.open("w", newline="") as ftrain, test_file.open("w", newline="") as ftest:
            train_writer = csv.writer(ftrain)
            test_writer = csv.writer(ftest)

            train_writer.writerow([header[i] for i in train_keep_idx])
            test_writer.writerow([header[i] for i in test_keep_idx])

            for row in reader:
                train_writer.writerow([row[i] if i < len(row) else "" for i in train_keep_idx])
                test_writer.writerow([row[i] if i < len(row) else "" for i in test_keep_idx])

    print(
        f"[OK] {input_file.name}: "
        f"train individuals={len(train_idx)}, test individuals={len(test_idx)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split column-oriented expression CSV files into train/test by individual IDs."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing CSV files to split."
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File pattern to process. Default: *.csv"
    )

    args = parser.parse_args()

    input_folder = args.input_folder.resolve()

    if not input_folder.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_folder}")

    train_dir = input_folder.with_name(input_folder.name + "_train")
    test_dir = input_folder.with_name(input_folder.name + "_test")

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    test_ids = {str(x).strip() for x in INDIVIDUAL_IDS}

    files = sorted(input_folder.glob(args.pattern))

    if not files:
        print(f"[WARN] No files found in {input_folder} with pattern {args.pattern}")
        return

    print(f"Input folder: {input_folder}")
    print(f"Train folder: {train_dir}")
    print(f"Test folder:  {test_dir}")
    print(f"Number of predefined test IDs: {len(test_ids)}")
    print(f"Number of files: {len(files)}")
    print()

    for file in files:
        split_file(file, train_dir, test_dir, test_ids)


if __name__ == "__main__":
    main()