#!/usr/bin/env bash

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input.gtf> <ENSEMBL_GENE_ID_1> [ENSEMBL_GENE_ID_2 ... ENSEMBL_GENE_ID_N]"
    exit 1
fi

INPUT_FILE="$1"
shift                     # removes GTF file from args
GENE_IDS=("$@")           # all remaining args are gene IDs

WINDOW=98304              # 196,608 kb window centered on TSS, as in Enformer

# convert bash array into AWK regex => id1|id2|id3
GENE_REGEX=$(printf "%s|" "${GENE_IDS[@]}")
GENE_REGEX=${GENE_REGEX%|}   # remove trailing |

awk -F'\t' -v genes="$GENE_REGEX" -v window="$WINDOW" '
BEGIN {
    pattern = "(" genes ")"
}
$3=="gene" {
    OFS = "\t"

    # TSS depends on strand... start or end
    if ($7 == "+") tss = $4
    else           tss = $5

    gene_id = "NA"
    if (match($9, /gene_id "([^"]+)"/, a)) gene_id = a[1]

    chrom = $1

    if (gene_id ~ pattern) {
        print chrom, gene_id, tss-window, tss+window
    }
}' "$INPUT_FILE"
