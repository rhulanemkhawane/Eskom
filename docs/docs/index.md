# Eskom Energy Demand Forecasting

This project predicts South Africa's hourly electricity demand using Eskom's published historical demand data.

## Overview

The pipeline covers end-to-end forecasting:
- Data ingestion and preprocessing from the [Eskom Data Portal](https://www.eskom.co.za/dataportal/)
- Feature engineering (calendar features, lag/rolling features)
- Rolling-origin cross-validation with multiple models (ETS, ElasticNet, HistGB, Naive baselines)
- Leakage-safe recursive multi-step forecasting

## Quick Start

See the [project README](https://github.com/rhulanemkhawane/eskom) for full setup and run instructions.

```bash
make requirements   # install dependencies
make data           # build processed dataset
make train          # run cross-validation
make predict        # generate predictions
```

## Project Structure

Source code lives in `src/eskom_energy_demand_forecasting/`. Configuration is in `configs/config.yaml`.

## Building These Docs

```bash
mkdocs build    # build static site
mkdocs serve    # serve locally at http://127.0.0.1:8000
```
