# TechSight — Trend Score Documentation

## Overview

The Technology Trend Score is a composite metric (0–100) that measures
how strongly a technology topic is trending based on patent and research
publication activity. It combines five components with configurable weights.

## Components

| Component | Weight | What it measures |
|---|---:|---|
| Growth Rate | 35% | Year-over-year change in total activity (patents + papers) |
| Consistency | 25% | How many recent years showed positive growth (not a one-time spike) |
| Acceleration | 20% | Whether growth is speeding up or slowing down |
| Activity Volume | 10% | Total activity level, log-scaled to avoid domination by large topics |
| Citation Signal | 10% | Average citation trend, when data is sufficiently available |

## Formulas

*To be filled in during Stage 5 implementation. Each formula will be
documented here with worked examples.*

### Growth Rate
TBD

### Consistency
TBD

### Acceleration
TBD

### Activity Volume
TBD

### Citation Signal
TBD

## Scaling

Each component is min-max scaled to 0–100 across all comparable topics,
then combined using the configured weights.

## Labels

| Label | Score Range | Condition |
|---|---|---|
| Emerging | 75–100 | Strong, consistent growth |
| Growing | 55–75 | Moderate positive trend |
| Stable | 40–55 | Flat or very slow change |
| Declining | 0–40 | Negative trend |
| Unusual Spike | — | Single year >3× median but no surrounding confirmation |

## Edge Cases

- **Insufficient history:** If fewer than 3 years of data exist, the score
  is marked as "low confidence" and no label is assigned.
- **Zero activity years:** Missing years within the range are treated as zero
  activity, which reduces consistency and growth scores.
- **Minimum activity threshold:** Topics with fewer than 10 total records
  in the lookback period do not receive a score (prevents noise).
- **Citation redistribution:** If citation data covers <50% of documents
  in a topic, the 10% citation weight is redistributed proportionally
  to the other four components. This is disclosed in the dashboard.

## Configuration

All weights and thresholds are in `config.py` under `TREND_SCORE_WEIGHTS`
and related constants.

---

*Exact formulas and worked examples will be added in Stage 5.*
