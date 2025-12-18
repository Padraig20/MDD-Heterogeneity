#!/usr/bin/env bash

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input.gtf> <ENSEMBL_GENE_ID_1> [ENSEMBL_GENE_ID_2 ... ENSEMBL_GENE_ID_N]"
    exit 1
fi

INPUT_FILE="$1"
shift                     # removes GTF file from args
GENE_IDS=("$@")           # all remaining args are gene IDs

WINDOW=98304              # 196,608 kb window centered on TSS, as in Enformer

# feed gene IDs to awk via STDIN (not as a giant regex, will kill process)
printf "%s\n" "${GENE_IDS[@]}" | awk -F'\t' -v window="$WINDOW" '
BEGIN { OFS="\t" }

# stdin => build a set
NR==FNR {
  gsub(/\r$/, "", $0)          # tolerate CRLF
  if ($0 != "") ids[$0] = 1
  next
}

# handle gtf
$3 == "gene" {
  # TSS depends on strand... start or end
  tss = ($7 == "+") ? $4 : $5

  gene_id = "NA"
  if (match($9, /gene_id "([^"]+)"/, a)) gene_id = a[1]
  else next

  if (gene_id in ids) {
    start = tss - window
    end   = tss + window
    print $1, gene_id, start, end
  }
}
' - "$INPUT_FILE"
