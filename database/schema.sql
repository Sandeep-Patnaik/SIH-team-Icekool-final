-- OceanMind AI — Database Schema (Module 2)
-- Fallback single source of truth if Alembic migrations aren't finished in time.
-- Kept in sync with database/models.py. Table/column names must never be renamed —
-- Modules 1, 4, 5, 6 all code against these exact names.

CREATE TABLE IF NOT EXISTS floats (
    float_id        VARCHAR PRIMARY KEY,
    deployment_lat  FLOAT,
    deployment_lon  FLOAT,
    deployment_date DATE,
    status          VARCHAR
);

CREATE TABLE IF NOT EXISTS profiles (
    id            SERIAL PRIMARY KEY,
    float_id      VARCHAR REFERENCES floats(float_id),
    cycle_number  INTEGER,
    profile_date  TIMESTAMP,
    latitude      FLOAT,
    longitude     FLOAT,
    ocean_region  VARCHAR
);

-- Upsert key for ingestion re-runs: one profile per (float_id, cycle_number).
CREATE UNIQUE INDEX IF NOT EXISTS ux_profiles_float_cycle
    ON profiles (float_id, cycle_number);

CREATE TABLE IF NOT EXISTS measurements (
    id               SERIAL PRIMARY KEY,
    profile_id       INTEGER REFERENCES profiles(id),
    pressure_dbar    FLOAT,
    depth_m          FLOAT,
    temperature_c    FLOAT,
    salinity_psu     FLOAT,
    dissolved_oxygen FLOAT,
    chlorophyll      FLOAT,
    ph               FLOAT,
    qc_flag          SMALLINT
);

CREATE INDEX IF NOT EXISTS ix_measurements_profile_id ON measurements (profile_id);

CREATE TABLE IF NOT EXISTS reports (
    id            SERIAL PRIMARY KEY,
    generated_at  TIMESTAMP,
    ocean_region  VARCHAR,
    period_start  DATE,
    period_end    DATE,
    file_path     VARCHAR,
    summary_text  TEXT
);

-- Helpful for Module 5/6 filtering by region + date range.
CREATE INDEX IF NOT EXISTS ix_profiles_region_date ON profiles (ocean_region, profile_date);
