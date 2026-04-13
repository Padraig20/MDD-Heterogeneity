#!/usr/bin/env bash

FILTER_PC=0

# optional flag
if [[ "$1" == "--protein-coding" ]]; then
    FILTER_PC=1
    shift
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [--protein-coding] <input.gtf> <ENSEMBL_GENE_ID_1> [...]"
    exit 1
fi

INPUT_FILE="$1"
shift
GENE_IDS=("$@")

WINDOW=98304

printf "%s\n" "${GENE_IDS[@]}" | awk -F'\t' -v window="$WINDOW" -v filter_pc="$FILTER_PC" '
BEGIN { OFS="\t" }

NR==FNR {
  gsub(/\r$/, "", $0)
  if ($0 != "") ids[$0] = 1
  next
}

$3 == "gene" {

  # --- optional protein-coding filter ---
  if (filter_pc) {
    if ($9 !~ /gene_type "protein_coding"/ && $9 !~ /gene_biotype "protein_coding"/) {
      next
    }
  }

  # TSS depends on strand
  if ($7 == "+") {
    tss = $4
  } else {
    tss = $5
  }

  if (match($9, /gene_id "([^"]+)"/)) {
    gene_id_start = index($9, "gene_id \"") + 9
    gene_id_substr = substr($9, gene_id_start)
    gene_id_end = index(gene_id_substr, "\"")
    gene_id = substr(gene_id_substr, 1, gene_id_end - 1)

    if (gene_id in ids) {
      start = tss - window - 1
      end   = tss + window - 1
      print $1, gene_id, tss, start, end
    }
  }
}
' - "$INPUT_FILE"