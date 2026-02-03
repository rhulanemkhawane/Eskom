# Eskom Energy Demand Forecasting

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project aims to accurately predict the South Africa's hourly energy demand using:

- **Eskom historical electricity demand data**
- **Weather data from Meteostat (temperature, wind, humidity, etc.)**

The goal is to **predict short-term electricity demand (hourly)** to support:
- Energy production planning
- Grid stability analysis
- Renewable energy integration
- Scenario and sensitivity analysis

This repository demonstrates an **end-to-end data science workflow**, from raw data ingestion to model evaluation, structured to professional standards.

---

## Business Motivation (Why This Project Matters)

Electricity demand forecasting is **critical** in South Africa due to:
- Load shedding risk
- High variability from weather
- Increasing renewable energy penetration
- Operational and financial planning requirements

Accurate hourly forecasts allow:
- Better dispatch planning
- Reduced operating costs
- Improved grid reliability
- Data-driven decision-making for utilities and energy traders

---

## Business Motivation (Why This Project Matters)

Electricity demand forecasting is **critical** in South Africa due to:
- Load shedding risk
- High variability from weather
- Increasing renewable energy penetration
- Operational and financial planning requirements

Accurate hourly forecasts allow:
- Better dispatch planning
- Reduced operating costs
- Improved grid reliability
- Data-driven decision-making for utilities and energy traders

---

## Data Sources

### 1. Eskom Electricity Demand Data
- Source: **:contentReference[oaicite:0]{index=0}**
- Description:
  - Historical national electricity demand
  - Hourly or sub-hourly resolution (depending on release)
  - Aggregated at grid level

> Eskom is South Africa’s primary electricity utility and the authoritative source for national demand data.

### 2. Weather Data
- Source: **(https://dev.meteostat.net/python/api/meteostat.hourly?utm_source=chatgpt.com)**
- Variables used:
  - Air temperature
  - Wind speed
  - Relative humidity
  - Atmospheric pressure

Weather is a **first-order driver** of electricity demand due to:
- Heating and cooling usage
- Industrial sensitivity to temperature
- Seasonal consumption patterns

 **Why combine demand + weather?**  
Electricity demand is **not purely time-driven**. Weather explains a large portion of short-term demand variation that pure time-series models cannot capture alone.

---
## Project Organization

This repository follows a **cookiecutter-style data science layout** to mirror real-world production projects:

```
├── LICENSE            <- MIT Open-source license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from Eskom Data Portal and Meteostat.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-rm-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         eskom_energy_demand_forecasting and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── eskom_energy_demand_forecasting   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes eskom_energy_demand_forecasting a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
