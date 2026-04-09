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
  if ($7 == "+") {
    tss = $4
  } else {
    tss = $5
  }

  # Extract gene_id from attributes field
  if (match($9, /gene_id "([^"]+)"/)) {
    gene_id_start = index($9, "gene_id \"") + 9
    gene_id_substr = substr($9, gene_id_start)
    gene_id_end = index(gene_id_substr, "\"")
    gene_id = substr(gene_id_substr, 1, gene_id_end - 1)
    
    if (gene_id in ids) {
      start = tss - window - 1
      end   = tss + window - 1
      print $1, gene_id, start, end
    }
  }
}
' - "$INPUT_FILE"
