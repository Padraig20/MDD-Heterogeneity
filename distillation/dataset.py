import os

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from pathlib import Path

from bed_reader import open_bed

"""
Quick reminder: Make sure to convert the VCF files to BED/BIM/FAM via plink2.

Example command:

plink2 \
  --vcf ALL.chr1.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz \
  --make-bed \
  --out chr1

We assume here that the BIM files have been read into memory (that should fit!)
and are nicely put into a dict with keys according to their chromosome (e.g. "chr1").
"""

class GenotypeDataset(Dataset):
    
    def __init__(self, bims: dict[str], idx2ind: dict[np.ndarray], y: Path | pd.DataFrame, bim_dir: str = "", window_size=1_000_000, select_genes: Path | None = None):
        """
        Args:
            bims (dict[str]):             Dictionary of BIM files indexed by chromosome.
            idx2ind (dict[np.ndarray]):   Dictionary mapping chromosome to array of individual indices.
            y (Path | pd.DataFrame): Path to target labels file (csv) or a DataFrame.
            bim_dir (Path):               Directory containing the BIM files.
            window_size (int):            Size of the genomic window to consider around each TSS.
            select_genes (Path | None):   Path to a file containing a list of genes to select. If None, use all genes.
        """
        self.bims        = bims
        self.bim_dir     = bim_dir
        self.idx2ind     = idx2ind
        self.window_size = window_size
        self.select_genes = select_genes
        if isinstance(y, Path) or isinstance(y, str):
            self.y    = pd.read_csv(y)
        else:
            self.y    = y # assume it's already a DataFrame
        # we will slightly change the y format to make training easier...
        # we currently have per row: gene,chrom,tss,individual1,individual2,...
        # we will denormalize this to have one row per gene-individual pair
        # with columns: gene, chrom, tss, individual, expression
        if "individual" not in self.y.columns:
            self.y = self.y.melt(id_vars=["gene", "chrom", "tss"], var_name="individual", value_name="expression")
        
        # get all different genes
        self.genes = self.y["gene"].unique()
    
        if self.select_genes is not None:
            if self.select_genes == Path("random"):
                # select 227 genes at random (same number as in MDD gene list) for quick testing
                np.random.seed(42)
                if len(self.genes) < 227:
                    selected_genes = self.genes
                else:
                    selected_genes = np.random.choice(self.genes, size=227, replace=False)
            else:
                selected_genes = set(pd.read_csv(self.select_genes, sep="\t")["ENSID"])
            self.genes = np.array([g for g in self.genes if g in selected_genes])
            self.y     = self.y[self.y["gene"].isin(selected_genes)].copy()


    def split_by_chromosome(self, chroms: list[str]) -> Dataset:
        # filter y and chroms to only include rows with chrom in chroms
        y_filtered       = self.y[self.y["chrom"].isin(chroms)].copy()
        bims_filtered    = {chrom: bim for chrom, bim in self.bims.items() if chrom in chroms}
        idx2ind_filtered = {chrom: idx2ind for chrom, idx2ind in self.idx2ind.items() if chrom in chroms}
        return GenotypeDataset(bims_filtered, idx2ind_filtered, y_filtered, bim_dir=self.bim_dir, window_size=self.window_size, select_genes=self.select_genes)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        row        = self.y.iloc[idx]
        ensid      = row["gene"]
        chrom      = row["chrom"]
        tss        = row["tss"]
        individual = row["individual"]
        expression = row["expression"]

        # find window in BIM file for this chrom that contains tss
        bim    = self.bims[chrom]
        start  = tss - self.window_size
        end    = tss + self.window_size
        idx2ind        = self.idx2ind[chrom]
        individual_idx = np.where(idx2ind == individual)[0][0]

        mask = (bim["chrom"] == chrom[3:]) & (bim["bp"] >= start) & (bim["bp"] <= end)
        var_idx = np.flatnonzero(mask.to_numpy())

        with open_bed(os.path.join(self.bim_dir, f"{chrom}.bed")) as bed:
            x = bed.read(index=np.s_[individual_idx, var_idx], dtype="int8")

        x = torch.from_numpy(x).flatten().float() # convert to float for training
        y = torch.tensor(expression, dtype=torch.float32)

        return x, y
    
    def get_gene_matrix(self, gene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the full design matrix for one gene.
        Good for LR - we model genes and cell-types separately... unhappy :(
        We train one row of the whole weight matrix of the LR.

        Returns:
            X       : shape (n_individuals_for_gene, n_snps_in_window)
            y       : shape (n_individuals_for_gene,)
            snp_ids : shape (n_snps_in_window,)
        """
        gene_data  = self.y[self.y["gene"] == gene]
        if len(gene_data) == 0:
            raise ValueError(f"No data found for gene {gene}")
        chrom      = gene_data["chrom"].iloc[0]
        tss        = gene_data["tss"].iloc[0]

        # find window in BIM file for this chrom that contains tss
        bim     = self.bims[chrom]
        start   = tss - self.window_size
        end     = tss + self.window_size

        mask    = (bim["chrom"] == chrom[3:]) & (bim["bp"] >= start) & (bim["bp"] <= end)
        var_idx = np.flatnonzero(mask.to_numpy())
        snp_ids = bim.loc[mask, "snp"].astype(str).to_numpy()

        fam_order_ids   = self.idx2ind[chrom].astype(str)
        row_individuals = gene_data["individual"].astype(str).to_numpy()
        ind_to_idx      = {ind: i for i, ind in enumerate(fam_order_ids)} # individuals to BED/FAM row indices

        # which individuals do we keep?
        keep_mask = np.array([ind in ind_to_idx for ind in row_individuals], dtype=bool)
        if not np.all(keep_mask):
            gene_data = gene_data.loc[keep_mask].copy()
            row_individuals = gene_data["individual"].astype(str).to_numpy()
        individual_idx = np.array([ind_to_idx[ind] for ind in row_individuals], dtype=int)

        with open_bed(os.path.join(self.bim_dir, f"{chrom}.bed")) as bed:
            X = bed.read(index=np.s_[individual_idx, var_idx], dtype="int8")

        y = gene_data.sort_values("individual")["expression"].to_numpy()

        return X, y, snp_ids

if __name__ == "__main__":
    # example usage
    bim = pd.read_csv(
        "chr1.bim",
        sep=r"\s+",
        header=None,
        names=["chrom", "snp", "cm", "bp", "a1", "a2"],
        dtype={"chrom": str, "snp": str, "bp": np.int64},
    )
    bims    = {"chr1": bim}
    
    idx2ind_arr = pd.read_csv("chr1.fam", sep=r"\s+", header=None, usecols=[0, 1], names=["family_id", "individual_id"])
    idx2ind_arr = idx2ind_arr["individual_id"].to_numpy()
    idx2ind = {"chr1": idx2ind_arr}
    
    y_path  = Path("student-target/0.csv")

    dataset = GenotypeDataset(bims=bims, idx2ind=idx2ind, y=y_path, select_genes=Path("data/mdd_genes.tsv"))
    print(f"Dataset size: {len(dataset)}")

    print("Splitting dataset by chromosomes 1...")
    new_dataset = dataset.split_by_chromosome(['chr1'])

    print(f"New dataset size (chromosomes 1): {len(new_dataset)}")

    for i in range(min(5, len(new_dataset))):
        input_data, label = new_dataset[i]
        print(f"Input shape at index {i}: {input_data.shape}, Label: {label}")

    for i in range(min(5, len(new_dataset.genes))):
        gene = new_dataset.genes[i]
        X, y, snp_ids = new_dataset.get_gene_matrix(new_dataset.genes[i])
        print(f"Design matrix shape for gene {gene}: {X.shape}, y shape: {y.shape}, snp_ids shape: {snp_ids.shape}")
    
    print(new_dataset.y.head())