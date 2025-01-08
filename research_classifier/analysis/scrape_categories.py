import requests
from bs4 import BeautifulSoup

category_classes = {
    "cs": "Computer Science",
    "econ": "Economics",
    "eess": "Electrical Engineering and Systems Science",
    "math": "Mathematics",
    "astro-ph": "Physics - Astrophysics",
    "cond-mat": "Physics - Condensed Matter",
    "gr-qc": "Physics - General Relativity and Quantum Cosmology",
    "hep-ex": "Physics - High Energy Physics - Experiment",
    "hep-lat": "Physics - High Energy Physics - Lattice",
    "hep-ph": "Physics - High Energy Physics - Phenomenology",
    "hep-th": "Physics - High Energy Physics - Theory",
    "math-ph": "Physics - Mathematical Physics",
    "nlin": "Physics - Nonlinear Sciences",
    "nucl-ex": "Physics - Nuclear Experiment",
    "nucl-th": "Physics - Nuclear Theory",
    "physics": "Physics",
    "quant-ph": "Physics - Quantum Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
}

scraped_categories = {
    "cs.AI": "Computer Science - Artificial Intelligence",
    "cs.AR": "Computer Science - Hardware Architecture",
    "cs.CC": "Computer Science - Computational Complexity",
    "cs.CE": "Computer Science - Computational Engineering, Finance, and Science",
    "cs.CG": "Computer Science - Computational Geometry",
    "cs.CL": "Computer Science - Computation and Language",
    "cs.CR": "Computer Science - Cryptography and Security",
    "cs.CV": "Computer Science - Computer Vision and Pattern Recognition",
    "cs.CY": "Computer Science - Computers and Society",
    "cs.DB": "Computer Science - Databases",
    "cs.DC": "Computer Science - Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Computer Science - Digital Libraries",
    "cs.DM": "Computer Science - Discrete Mathematics",
    "cs.DS": "Computer Science - Data Structures and Algorithms",
    "cs.ET": "Computer Science - Emerging Technologies",
    "cs.FL": "Computer Science - Formal Languages and Automata Theory",
    "cs.GL": "Computer Science - General Literature",
    "cs.GR": "Computer Science - Graphics",
    "cs.GT": "Computer Science - Computer Science and Game Theory",
    "cs.HC": "Computer Science - Human-Computer Interaction",
    "cs.IR": "Computer Science - Information Retrieval",
    "cs.IT": "Computer Science - Information Theory",
    "cs.LG": "Computer Science - Machine Learning",
    "cs.LO": "Computer Science - Logic in Computer Science",
    "cs.MA": "Computer Science - Multiagent Systems",
    "cs.MM": "Computer Science - Multimedia",
    "cs.MS": "Computer Science - Mathematical Software",
    "cs.NA": "Computer Science - Numerical Analysis",
    "cs.NE": "Computer Science - Neural and Evolutionary Computing",
    "cs.NI": "Computer Science - Networking and Internet Architecture",
    "cs.OH": "Computer Science - Other Computer Science",
    "cs.OS": "Computer Science - Operating Systems",
    "cs.PF": "Computer Science - Performance",
    "cs.PL": "Computer Science - Programming Languages",
    "cs.RO": "Computer Science - Robotics",
    "cs.SC": "Computer Science - Symbolic Computation",
    "cs.SD": "Computer Science - Sound",
    "cs.SE": "Computer Science - Software Engineering",
    "cs.SI": "Computer Science - Social and Information Networks",
    "cs.SY": "Computer Science - Systems and Control",
    "econ.EM": "Economics - Econometrics",
    "econ.GN": "Economics - General Economics",
    "econ.TH": "Economics - Theoretical Economics",
    "eess.AS": "Electrical Engineering and Systems Science - Audio and Speech Processing",
    "eess.IV": "Electrical Engineering and Systems Science - Image and Video Processing",
    "eess.SP": "Electrical Engineering and Systems Science - Signal Processing",
    "eess.SY": "Electrical Engineering and Systems Science - Systems and Control",
    "math.AC": "Mathematics - Commutative Algebra",
    "math.AG": "Mathematics - Algebraic Geometry",
    "math.AP": "Mathematics - Analysis of PDEs",
    "math.AT": "Mathematics - Algebraic Topology",
    "math.CA": "Mathematics - Classical Analysis and ODEs",
    "math.CO": "Mathematics - Combinatorics",
    "math.CT": "Mathematics - Category Theory",
    "math.CV": "Mathematics - Complex Variables",
    "math.DG": "Mathematics - Differential Geometry",
    "math.DS": "Mathematics - Dynamical Systems",
    "math.FA": "Mathematics - Functional Analysis",
    "math.GM": "Mathematics - General Mathematics",
    "math.GN": "Mathematics - General Topology",
    "math.GR": "Mathematics - Group Theory",
    "math.GT": "Mathematics - Geometric Topology",
    "math.HO": "Mathematics - History and Overview",
    "math.IT": "Mathematics - Information Theory",
    "math.KT": "Mathematics - K-Theory and Homology",
    "math.LO": "Mathematics - Logic",
    "math.MG": "Mathematics - Metric Geometry",
    "math.MP": "Mathematics - Mathematical Physics",
    "math.NA": "Mathematics - Numerical Analysis",
    "math.NT": "Mathematics - Number Theory",
    "math.OA": "Mathematics - Operator Algebras",
    "math.OC": "Mathematics - Optimization and Control",
    "math.PR": "Mathematics - Probability",
    "math.QA": "Mathematics - Quantum Algebra",
    "math.RA": "Mathematics - Rings and Algebras",
    "math.RT": "Mathematics - Representation Theory",
    "math.SG": "Mathematics - Symplectic Geometry",
    "math.SP": "Mathematics - Spectral Theory",
    "math.ST": "Mathematics - Statistics Theory",
    "astro-ph.CO": "Physics - Astrophysics - Cosmology and Nongalactic Astrophysics",
    "astro-ph.EP": "Physics - Astrophysics - Earth and Planetary Astrophysics",
    "astro-ph.GA": "Physics - Astrophysics - Astrophysics of Galaxies",
    "astro-ph.HE": "Physics - Astrophysics - High Energy Astrophysical Phenomena",
    "astro-ph.IM": "Physics - Astrophysics - Instrumentation and Methods for Astrophysics",
    "astro-ph.SR": "Physics - Astrophysics - Solar and Stellar Astrophysics",
    "cond-mat.dis-nn": "Physics - Condensed Matter - Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Physics - Condensed Matter - Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Physics - Condensed Matter - Materials Science",
    "cond-mat.other": "Physics - Condensed Matter - Other Condensed Matter",
    "cond-mat.quant-gas": "Physics - Condensed Matter - Quantum Gases",
    "cond-mat.soft": "Physics - Condensed Matter - Soft Condensed Matter",
    "cond-mat.stat-mech": "Physics - Condensed Matter - Statistical Mechanics",
    "cond-mat.str-el": "Physics - Condensed Matter - Strongly Correlated Electrons",
    "cond-mat.supr-con": "Physics - Condensed Matter - Superconductivity",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nlin.AO": "Physics - Nonlinear Sciences - Adaptation and Self-Organizing Systems",
    "nlin.CD": "Physics - Nonlinear Sciences - Chaotic Dynamics",
    "nlin.CG": "Physics - Nonlinear Sciences - Cellular Automata and Lattice Gases",
    "nlin.PS": "Physics - Nonlinear Sciences - Pattern Formation and Solitons",
    "nlin.SI": "Physics - Nonlinear Sciences - Exactly Solvable and Integrable Systems",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics.acc-ph": "Physics - Accelerator Physics",
    "physics.ao-ph": "Physics - Atmospheric and Oceanic Physics",
    "physics.app-ph": "Physics - Applied Physics",
    "physics.atm-clus": "Physics - Atomic and Molecular Clusters",
    "physics.atom-ph": "Physics - Atomic Physics",
    "physics.bio-ph": "Physics - Biological Physics",
    "physics.chem-ph": "Physics - Chemical Physics",
    "physics.class-ph": "Physics - Classical Physics",
    "physics.comp-ph": "Physics - Computational Physics",
    "physics.data-an": "Physics - Data Analysis, Statistics and Probability",
    "physics.ed-ph": "Physics - Physics Education",
    "physics.flu-dyn": "Physics - Fluid Dynamics",
    "physics.gen-ph": "Physics - General Physics",
    "physics.geo-ph": "Physics - Geophysics",
    "physics.hist-ph": "Physics - History and Philosophy of Physics",
    "physics.ins-det": "Physics - Instrumentation and Detectors",
    "physics.med-ph": "Physics - Medical Physics",
    "physics.optics": "Physics - Optics",
    "physics.plasm-ph": "Physics - Plasma Physics",
    "physics.pop-ph": "Physics - Popular Physics",
    "physics.soc-ph": "Physics - Physics and Society",
    "physics.space-ph": "Physics - Space Physics",
    "quant-ph": "Quantum Physics",
    "q-bio.BM": "Quantitative Biology - Biomolecules",
    "q-bio.CB": "Quantitative Biology - Cell Behavior",
    "q-bio.GN": "Quantitative Biology - Genomics",
    "q-bio.MN": "Quantitative Biology - Molecular Networks",
    "q-bio.NC": "Quantitative Biology - Neurons and Cognition",
    "q-bio.OT": "Quantitative Biology - Other Quantitative Biology",
    "q-bio.PE": "Quantitative Biology - Populations and Evolution",
    "q-bio.QM": "Quantitative Biology - Quantitative Methods",
    "q-bio.SC": "Quantitative Biology - Subcellular Processes",
    "q-bio.TO": "Quantitative Biology - Tissues and Organs",
    "q-fin.CP": "Quantitative Finance - Computational Finance",
    "q-fin.EC": "Quantitative Finance - Economics",
    "q-fin.GN": "Quantitative Finance - General Finance",
    "q-fin.MF": "Quantitative Finance - Mathematical Finance",
    "q-fin.PM": "Quantitative Finance - Portfolio Management",
    "q-fin.PR": "Quantitative Finance - Pricing of Securities",
    "q-fin.RM": "Quantitative Finance - Risk Management",
    "q-fin.ST": "Quantitative Finance - Statistical Finance",
    "q-fin.TR": "Quantitative Finance - Trading and Market Microstructure",
    "stat.AP": "Statistics - Applications",
    "stat.CO": "Statistics - Computation",
    "stat.ME": "Statistics - Methodology",
    "stat.ML": "Statistics - Machine Learning",
    "stat.OT": "Statistics - Other Statistics",
    "stat.TH": "Statistics - Statistics Theory",
}

scraped_categories.update(category_classes)


# Send HTTP request to get the page content
def scrape_categories():
    # URL of arXiv category taxonomy
    url = "https://arxiv.org/category_taxonomy"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
    else:
        print("Failed to fetch the page")
        exit()

    # Find all categories in the taxonomy page
    categories = {}
    primary_sections = soup.find_all("h4")
    for section in primary_sections:
        tag = section.text.strip()
        if tag.startswith("Category Name"):
            continue
        tag = tag.split(" ")[0]
        name = section.find("span").text.strip()
        # remove brackets
        name = name[1:-1]

        categories[tag] = name

    for category, name in categories.items():
        segments = category.split(".")
        if len(segments) > 1:
            cat_class = segments[0]
            categories[category] = category_classes[cat_class] + " - " + name

    # include category classes too as some abstracts are tagged this way
    scraped_categories.update(category_classes)
    return scraped_categories


if __name__ == "__main__":
    categories = scrape_categories()
    print(categories)
