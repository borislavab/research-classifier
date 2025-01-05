import pandas as pd
from typing import List
import kagglehub
import os
from datasets import load_dataset, Dataset

LABELS = [
    "acc-phys",
    "adap-org",
    "alg-geom",
    "ao-sci",
    "astro-ph",
    "astro-ph.CO",
    "astro-ph.EP",
    "astro-ph.GA",
    "astro-ph.HE",
    "astro-ph.IM",
    "astro-ph.SR",
    "atom-ph",
    "bayes-an",
    "chao-dyn",
    "chem-ph",
    "cmp-lg",
    "comp-gas",
    "cond-mat",
    "cond-mat.dis-nn",
    "cond-mat.mes-hall",
    "cond-mat.mtrl-sci",
    "cond-mat.other",
    "cond-mat.quant-gas",
    "cond-mat.soft",
    "cond-mat.stat-mech",
    "cond-mat.str-el",
    "cond-mat.supr-con",
    "cs.AI",
    "cs.AR",
    "cs.CC",
    "cs.CE",
    "cs.CG",
    "cs.CL",
    "cs.CR",
    "cs.CV",
    "cs.CY",
    "cs.DB",
    "cs.DC",
    "cs.DL",
    "cs.DM",
    "cs.DS",
    "cs.ET",
    "cs.FL",
    "cs.GL",
    "cs.GR",
    "cs.GT",
    "cs.HC",
    "cs.IR",
    "cs.IT",
    "cs.LG",
    "cs.LO",
    "cs.MA",
    "cs.MM",
    "cs.MS",
    "cs.NA",
    "cs.NE",
    "cs.NI",
    "cs.OH",
    "cs.OS",
    "cs.PF",
    "cs.PL",
    "cs.RO",
    "cs.SC",
    "cs.SD",
    "cs.SE",
    "cs.SI",
    "cs.SY",
    "dg-ga",
    "econ.EM",
    "econ.GN",
    "econ.TH",
    "eess.AS",
    "eess.IV",
    "eess.SP",
    "eess.SY",
    "funct-an",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math-ph",
    "math.AC",
    "math.AG",
    "math.AP",
    "math.AT",
    "math.CA",
    "math.CO",
    "math.CT",
    "math.CV",
    "math.DG",
    "math.DS",
    "math.FA",
    "math.GM",
    "math.GN",
    "math.GR",
    "math.GT",
    "math.HO",
    "math.IT",
    "math.KT",
    "math.LO",
    "math.MG",
    "math.MP",
    "math.NA",
    "math.NT",
    "math.OA",
    "math.OC",
    "math.PR",
    "math.QA",
    "math.RA",
    "math.RT",
    "math.SG",
    "math.SP",
    "math.ST",
    "mtrl-th",
    "nlin.AO",
    "nlin.CD",
    "nlin.CG",
    "nlin.PS",
    "nlin.SI",
    "nucl-ex",
    "nucl-th",
    "patt-sol",
    "physics.acc-ph",
    "physics.ao-ph",
    "physics.app-ph",
    "physics.atm-clus",
    "physics.atom-ph",
    "physics.bio-ph",
    "physics.chem-ph",
    "physics.class-ph",
    "physics.comp-ph",
    "physics.data-an",
    "physics.ed-ph",
    "physics.flu-dyn",
    "physics.gen-ph",
    "physics.geo-ph",
    "physics.hist-ph",
    "physics.ins-det",
    "physics.med-ph",
    "physics.optics",
    "physics.plasm-ph",
    "physics.pop-ph",
    "physics.soc-ph",
    "physics.space-ph",
    "plasm-ph",
    "q-alg",
    "q-bio",
    "q-bio.BM",
    "q-bio.CB",
    "q-bio.GN",
    "q-bio.MN",
    "q-bio.NC",
    "q-bio.OT",
    "q-bio.PE",
    "q-bio.QM",
    "q-bio.SC",
    "q-bio.TO",
    "q-fin.CP",
    "q-fin.EC",
    "q-fin.GN",
    "q-fin.MF",
    "q-fin.PM",
    "q-fin.PR",
    "q-fin.RM",
    "q-fin.ST",
    "q-fin.TR",
    "quant-ph",
    "solv-int",
    "stat.AP",
    "stat.CO",
    "stat.ME",
    "stat.ML",
    "stat.OT",
    "stat.TH",
    "supr-con",
]


def download():
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    print("Path to dataset files:", path)
    return os.path.join(path, "arxiv-metadata-oai-snapshot.json")


def load(dataset_path: str = None, head: int = None) -> Dataset:
    if not dataset_path:
        dataset_path = download()
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.select_columns(["title", "categories", "abstract"])
    if head:
        return dataset.select(range(head))
    return dataset


def extract_categories(dataset: Dataset) -> List[str]:
    all_categories = set()
    for categories in dataset["categories"]:
        all_categories.update(categories.split(" "))
    return sorted(all_categories)


def extract_labels(categories: str) -> List[int]:
    sample_categories = set(categories.split(" "))
    return [1.0 if label in sample_categories else 0.0 for label in LABELS]


if __name__ == "__main__":
    path = download()
    dataset = load(path)
    LABELS = extract_categories(dataset)
    print(LABELS)
