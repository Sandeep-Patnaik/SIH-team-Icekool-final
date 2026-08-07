# 🌊 OceanMind AI - Project Structure

## Project Overview

OceanMind AI is an AI-powered Ocean Intelligence Platform developed for the Smart India Hackathon (SIH).

The objective is to transform complex ARGO oceanographic datasets into an intelligent platform that enables natural language interaction, interactive visualizations, AI-generated insights, and decision support for researchers, students, policymakers, and marine organizations.

---

# 🏗️ Overall System Architecture

Official ARGO NetCDF Files

↓

Data Ingestion & Processing

↓

PostgreSQL Database

↓

Vector Database (FAISS / ChromaDB)

↓

RAG Pipeline + LLM

↓

Ocean Intelligence Engine

↓

Interactive Dashboard & AI Assistant

---

# 💻 Technology Stack

## Frontend
- Streamlit
- Plotly
- Folium / Leaflet

## Backend
- Python
- PostgreSQL

## AI
- LangChain
- FAISS / ChromaDB
- OpenAI / Qwen / Llama
- MCP (Future)

## Data Processing
- Pandas
- Xarray
- NetCDF4
- NumPy

---

# 👥 Team Modules

| Member | Module |
|---------|----------------------------|
| Member 1 | Data Engineering & Database |
| Member 2 | Dashboard & Visualization |
| Member 3 | AI (RAG + Chatbot) |
| Member 4 | Ocean Intelligence Engine |
| Member 5 | Reports & Export |
| Member 6 | Integration, Testing & Deployment |

---

# 📂 Temporary Development Structure (During Development)

Each member develops independently inside their own folder using their own Git branch.

These folders are **temporary** and exist only during development to avoid merge conflicts.

```text
OceanMind-AI/

member1_data_engineering/      (Temporary)

member2_dashboard/             (Temporary)

member3_ai/                    (Temporary)

member4_intelligence/          (Temporary)

member5_reports/               (Temporary)

member6_integration/           (Temporary)
```

Each member is responsible only for their assigned folder.

After module completion, code will be reviewed, tested, and merged into the final project structure.

---

# 📂 Final Integrated Project Structure

```text
OceanMind-AI/

│
├── ai/
│   ├── chatbot.py
│   ├── rag.py
│   ├── embeddings.py
│   ├── memory.py
│   ├── prompts.py
│   └── sql_agent.py
│
├── ingestion/
│   ├── parser.py
│   ├── loader.py
│   ├── database.py
│   ├── config.py
│   └── utils.py
│
├── frontend/
│   ├── dashboard.py
│   ├── maps.py
│   ├── charts.py
│   ├── components.py
│   └── theme.py
│
├── intelligence/
│   ├── health.py
│   ├── insights.py
│   ├── recommendation.py
│   └── summary.py
│
├── reports/
│   ├── report_generator.py
│   ├── pdf.py
│   ├── export.py
│   └── templates.py
│
├── database/
│
├── utils/
│
├── data/
│
├── tests/
│
├── docs/
│
├── main.py
│
├── requirements.txt
│
├── README.md
│
└── PROJECT_STRUCTURE.md
```

---

# 🔄 Development Workflow

Official ARGO Dataset

↓

Data Ingestion

↓

PostgreSQL

↓

Dashboard

↓

AI (RAG)

↓

Ocean Intelligence

↓

Reports

↓

Testing

↓

Final Integration

↓

Deployment

---

# 🌿 Git Branch Strategy

```text
main

feature/data-engineering

feature/dashboard

feature/ai

feature/intelligence

feature/reports

feature/integration
```

Each member works only on their own feature branch.

No direct commits to `main`.

---

# 🔗 Integration Order

1. Data Ingestion
2. PostgreSQL
3. Dashboard
4. AI (RAG + Chatbot)
5. Ocean Intelligence
6. Reports
7. Final Testing
8. Deployment

---

# 🚀 Future Scope

- NOAA Integration
- INCOIS Integration
- ISRO Integration
- Satellite Data
- Deep ARGO
- BGC ARGO
- Multi-Agent AI
- Ocean Forecasting
- Ocean Risk Prediction