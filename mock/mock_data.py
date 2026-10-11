"""
mock_data.py — Generate realistic mock data for TechSight.

Creates synthetic patent and research paper records for demonstration
and testing purposes. All data is clearly labeled as MOCK.

CRITICAL: This is MOCK DATA — never present it as real research.
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Realistic titles and abstracts for each domain
# ---------------------------------------------------------------------------

SSB_TITLES = [
    "Novel Sulfide-Based Solid Electrolyte for All-Solid-State Lithium Batteries",
    "Interface Engineering of Garnet-Type Solid Electrolytes",
    "High-Performance Solid-State Battery with Lithium Metal Anode",
    "Scalable Manufacturing of Thin-Film Solid Electrolytes",
    "Composite Cathode Design for Solid-State Lithium-Ion Batteries",
    "Argyrodite Li6PS5Cl Solid Electrolyte with Enhanced Ionic Conductivity",
    "In-Situ Formation of Stable Interface in Solid-State Batteries",
    "Oxide-Based Solid Electrolytes: A Comparative Study",
    "Polymer-Ceramic Composite Electrolytes for Flexible Solid-State Batteries",
    "Dendrite Suppression Mechanisms in Solid-State Lithium Metal Batteries",
    "Room-Temperature Solid-State Battery Using NASICON Electrolyte",
    "Mechanical Properties of Sulfide Solid Electrolytes Under Pressure",
    "Solid-State Electrolyte Membrane for Grid-Scale Energy Storage",
    "Li7La3Zr2O12 Garnet Electrolyte with Aluminum Doping",
    "Solid-State Battery Pack Design for Electric Vehicle Applications",
]

QC_TITLES = [
    "Quantum Error Correction with Surface Codes on Superconducting Qubits",
    "Variational Quantum Eigensolver for Molecular Simulation",
    "Quantum Approximate Optimization Algorithm for Combinatorial Problems",
    "Noise-Resilient Quantum Gate Design Using Optimal Control Theory",
    "Topological Quantum Computing with Majorana Fermions",
    "Quantum Machine Learning for Classification Tasks",
    "Scalable Ion-Trap Quantum Computer Architecture",
    "Quantum Random Walk Algorithms for Graph Problems",
    "Fault-Tolerant Quantum Computing with Logical Qubits",
    "Quantum Supremacy Experiments on 100+ Qubit Processors",
    "Hybrid Classical-Quantum Algorithms for Optimization",
    "Quantum Key Distribution Protocol with Enhanced Security",
    "Quantum Annealing for Protein Folding Optimization",
    "Photonic Quantum Computing with Squeezed States of Light",
    "Quantum Entanglement Distribution Over Fiber Networks",
]

GAI_TITLES = [
    "Scaling Laws for Large Language Models: A Comprehensive Study",
    "Diffusion Models for High-Resolution Image Synthesis",
    "Instruction Tuning and RLHF for Aligned Language Models",
    "Efficient Fine-Tuning with Low-Rank Adaptation (LoRA)",
    "Multimodal Foundation Models: Vision-Language Integration",
    "Text-to-Video Generation with Temporal Consistency",
    "Retrieval-Augmented Generation for Knowledge-Intensive Tasks",
    "Constitutional AI: Training Helpful and Harmless Assistants",
    "Generative Adversarial Networks for Medical Image Synthesis",
    "Transformer Architecture Improvements for Long-Context Windows",
    "Prompt Engineering Techniques for Zero-Shot Learning",
    "Generative AI for Drug Discovery and Molecular Design",
    "Code Generation with Large Language Models: Benchmarks and Safety",
    "Latent Diffusion Models for Text-to-Image Generation",
    "Mixture of Experts for Efficient Large-Scale Language Models",
]

DOMAIN_TITLES = {
    "solid_state_batteries": SSB_TITLES,
    "quantum_computing": QC_TITLES,
    "generative_ai": GAI_TITLES,
}

ORGS = [
    "MIT", "Stanford University", "Tsinghua University", "KAIST",
    "University of Tokyo", "ETH Zurich", "Caltech", "UC Berkeley",
    "Max Planck Institute", "Cambridge University", "IBM Research",
    "Google Research", "Microsoft Research", "Samsung SDI",
    "Toyota Research Institute", "CATL", "QuantumScape",
]

AUTHOR_FIRST = ["Wei", "Jia", "Li", "James", "Sarah", "Chen", "Priya",
                "Alex", "Maria", "Yuki", "Hans", "Fatima", "Robert"]
AUTHOR_LAST = ["Zhang", "Wang", "Li", "Smith", "Johnson", "Kim", "Patel",
               "Mueller", "Garcia", "Tanaka", "Chen", "Ahmed", "Wilson"]

# ---------------------------------------------------------------------------
# Growth patterns — realistic trends for each domain
# ---------------------------------------------------------------------------

def _yearly_counts():
    """Return realistic publication counts per domain per year (2012-2024)."""
    years = list(range(2012, 2025))

    # Solid-state batteries: steady growth
    ssb_papers  = [35, 42, 51, 60, 75, 95, 120, 150, 185, 230, 280, 340, 370]
    ssb_patents = [12, 15, 18, 22, 28, 35, 45,  58,  72,  90, 110, 135, 125]

    # Quantum computing: accelerating growth
    qc_papers   = [80, 95, 110, 130, 155, 190, 240, 310, 400, 520, 650, 800, 900]
    qc_patents  = [10, 12, 15,  18,  25,  35,  50,  70,  95, 130, 175, 230, 260]

    # Generative AI: exponential recent growth
    gai_papers  = [20, 25, 30, 35, 42, 55, 80, 130, 250, 500, 1000, 1800, 2500]
    gai_patents = [ 5,  6,  8, 10, 12, 18, 30,  55, 100, 200,  400,  700,  950]

    return {
        "solid_state_batteries": {"years": years, "papers": ssb_papers, "patents": ssb_patents},
        "quantum_computing":     {"years": years, "papers": qc_papers,  "patents": qc_patents},
        "generative_ai":         {"years": years, "papers": gai_papers, "patents": gai_patents},
    }


def _random_author():
    first = random.choice(AUTHOR_FIRST)
    last = random.choice(AUTHOR_LAST)
    return f"{first} {last}"


def _random_authors():
    n = random.randint(1, 5)
    return "; ".join([_random_author() for _ in range(n)])


def generate_documents(max_per_domain=200):
    """
    Generate mock documents for all three technology domains.

    Returns a pandas DataFrame matching the 'documents' table schema.
    """
    records = []
    doc_id = 1
    counts = _yearly_counts()

    for domain, titles in DOMAIN_TITLES.items():
        domain_counts = counts[domain]
        years = domain_counts["years"]
        paper_totals = domain_counts["papers"]
        patent_totals = domain_counts["patents"]

        # Scale down to fit max_per_domain
        total = sum(paper_totals) + sum(patent_totals)
        scale = min(1.0, max_per_domain / total) if total > 0 else 1.0

        for i, year in enumerate(years):
            n_papers = max(1, int(paper_totals[i] * scale))
            n_patents = max(1, int(patent_totals[i] * scale))

            for _ in range(n_papers):
                title = random.choice(titles)
                records.append({
                    "id": doc_id,
                    "source": "openalex",
                    "source_id": f"W{random.randint(1000000000, 9999999999)}",
                    "title": f"[MOCK] {title}",
                    "abstract": f"[MOCK DATA] This paper investigates {title.lower()}. "
                                f"Our approach demonstrates significant improvements in the field.",
                    "year": year,
                    "classifications": domain.replace("_", " "),
                    "authors": _random_authors(),
                    "organizations": "; ".join(random.sample(ORGS, random.randint(1, 3))),
                    "citation_count": max(0, int(np.random.lognormal(2, 1.5)) - (2025 - year)),
                    "url": f"https://openalex.org/W{random.randint(1000000, 9999999)}",
                    "domain": domain,
                    "ingested_at": datetime(year, random.randint(1, 12), random.randint(1, 28)),
                })
                doc_id += 1

            for _ in range(n_patents):
                title = random.choice(titles)
                records.append({
                    "id": doc_id,
                    "source": "uspto",
                    "source_id": f"US{random.randint(10000000, 99999999)}",
                    "title": f"[MOCK] {title}",
                    "abstract": f"[MOCK DATA] A method and apparatus for {title.lower()}.",
                    "year": year,
                    "classifications": domain.replace("_", " "),
                    "authors": _random_authors(),
                    "organizations": random.choice(ORGS),
                    "citation_count": None,  # USPTO citations often unavailable
                    "url": f"https://patents.google.com/patent/US{random.randint(10000000, 99999999)}",
                    "domain": domain,
                    "ingested_at": datetime(year, random.randint(1, 12), random.randint(1, 28)),
                })
                doc_id += 1

    return pd.DataFrame(records)


def get_yearly_indicators():
    """
    Return pre-computed yearly indicators per domain (patent count, paper count, etc.).

    Returns a DataFrame matching the technology_year_indicators schema.
    """
    counts = _yearly_counts()
    rows = []
    tech_id = 1
    for domain_name, data in counts.items():
        for i, year in enumerate(data["years"]):
            papers = data["papers"][i]
            patents = data["patents"][i]
            rows.append({
                "technology_id": tech_id,
                "technology_name": "Generative AI" if domain_name == "generative_ai" else domain_name.replace("_", " ").title(),
                "domain": domain_name,
                "year": year,
                "patent_count": patents,
                "paper_count": papers,
                "total_count": papers + patents,
                "avg_citations": round(np.random.lognormal(2, 0.8), 1),
                "is_complete": year < 2024,
            })
        tech_id += 1

    return pd.DataFrame(rows)


def get_technologies():
    """Return a DataFrame of discovered technology topics."""
    techs = [
        {"id": 1, "name": "Solid State Batteries",
         "cluster_id": 0,
         "top_terms": "solid electrolyte, lithium, ionic conductivity, garnet, sulfide",
         "document_count": 520,
         "description": "Research and patents on solid-state battery electrolytes and systems"},
        {"id": 2, "name": "Quantum Computing",
         "cluster_id": 1,
         "top_terms": "qubit, quantum gate, error correction, entanglement, superposition",
         "document_count": 780,
         "description": "Quantum computing hardware, algorithms, and error correction"},
        {"id": 3, "name": "Generative AI",
         "cluster_id": 2,
         "top_terms": "language model, transformer, diffusion, generation, fine-tuning",
         "document_count": 1450,
         "description": "Large language models, diffusion models, and generative AI systems"},
    ]
    return pd.DataFrame(techs)


def get_trend_scores():
    """Return pre-computed trend scores for each technology."""
    scores = [
        {
            "technology_id": 1,
            "technology_name": "Solid State Batteries",
            "score": 72.4,
            "label": "Growing",
            "growth_rate": 78.5,
            "consistency": 85.0,
            "acceleration": 45.2,
            "activity_volume": 65.0,
            "citation_signal": 70.0,
        },
        {
            "technology_id": 2,
            "technology_name": "Quantum Computing",
            "score": 82.1,
            "label": "Emerging",
            "growth_rate": 88.2,
            "consistency": 90.0,
            "acceleration": 72.5,
            "activity_volume": 78.0,
            "citation_signal": 65.0,
        },
        {
            "technology_id": 3,
            "technology_name": "Generative AI",
            "score": 96.8,
            "label": "Emerging",
            "growth_rate": 99.5,
            "consistency": 95.0,
            "acceleration": 98.0,
            "activity_volume": 95.0,
            "citation_signal": 88.0,
        },
    ]
    return pd.DataFrame(scores)


def get_forecasts():
    """Return forecast predictions for 2025-2027."""
    counts = _yearly_counts()
    forecasts = []

    for domain_name, data in counts.items():
        tech_name = "Generative AI" if domain_name == "generative_ai" else domain_name.replace("_", " ").title()
        papers = data["papers"]
        patents = data["patents"]

        # Simple linear extrapolation for mock forecasts
        for series, values in [("paper_count", papers), ("patent_count", patents)]:
            recent_growth = (values[-1] - values[-3]) / 2
            last_val = values[-1]
            for j, year in enumerate([2025, 2026, 2027]):
                pred = last_val + recent_growth * (j + 1)
                forecasts.append({
                    "technology_name": tech_name,
                    "domain": domain_name,
                    "year": year,
                    "series_type": series,
                    "predicted_value": round(max(0, pred)),
                    "lower_bound": round(max(0, pred * 0.8)),
                    "upper_bound": round(pred * 1.2),
                    "is_future": True,
                })

    return pd.DataFrame(forecasts)


if __name__ == "__main__":
    print("Generating mock data...")
    docs = generate_documents()
    print(f"  Documents: {len(docs)} records")
    indicators = get_yearly_indicators()
    print(f"  Indicators: {len(indicators)} records")
    techs = get_technologies()
    print(f"  Technologies: {len(techs)} records")
    scores = get_trend_scores()
    print(f"  Trend Scores: {len(scores)} records")
    forecasts = get_forecasts()
    print(f"  Forecasts: {len(forecasts)} records")
    print("Done.")
