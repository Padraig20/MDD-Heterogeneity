# Usage: ./get_gene_tss.sh <input.gtf> <ENSEMBL_GENE_ID>

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input.gtf> <ENSEMBL_GENE_ID>"
    exit 1
fi

INPUT_FILE="$1"
GENE_ID_SEARCH="$2"
WINDOW=98304 # 196,608 kb window centered on TSS, as in Enformer

awk -F'\t' -v gene_search="$GENE_ID_SEARCH" -v window="$WINDOW" '
$3=="gene" {
    OFS = "\t"

    if ($7 == "+") tss = $4
    else           tss = $5

    gene_id = "NA"
    if (match($9, /gene_id "([^"]+)"/, a)) {
        gene_id = a[1]
    }

    if (gene_id == gene_search) {
        print gene_id, tss-window, tss+window
    }
}' "$INPUT_FILE"
