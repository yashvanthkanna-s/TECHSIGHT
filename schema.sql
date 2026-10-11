-- ============================================================
-- TechSight — PostgreSQL Database Schema
-- ============================================================
-- Run with:  psql -U postgres -d techsight -f schema.sql
--
-- This creates all tables from scratch. If tables already exist,
-- they will be dropped and recreated (data loss!).
-- In production you would use migrations; for this student project
-- we use a single schema file for simplicity.
-- ============================================================

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS forecasts CASCADE;
DROP TABLE IF EXISTS forecast_runs CASCADE;
DROP TABLE IF EXISTS trend_scores CASCADE;
DROP TABLE IF EXISTS technology_year_indicators CASCADE;
DROP TABLE IF EXISTS document_topics CASCADE;
DROP TABLE IF EXISTS technologies CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS ingestion_log CASCADE;

-- ============================================================
-- 1. documents — All patents and research papers in common format
-- ============================================================
-- Each row is one document from one source.
-- source_id is the identifier from the original source (e.g. OpenAlex work ID).
CREATE TABLE documents (
    id              SERIAL PRIMARY KEY,
    source          VARCHAR(50)   NOT NULL,  -- 'openalex' or 'uspto'
    source_id       VARCHAR(255)  NOT NULL,  -- original ID from the source
    title           TEXT          NOT NULL,
    abstract        TEXT,                    -- may be null if unavailable
    year            INTEGER       NOT NULL,
    classifications TEXT,                    -- semicolon-separated subject areas
    authors         TEXT,                    -- semicolon-separated author names
    organizations   TEXT,                    -- semicolon-separated affiliations
    citation_count  INTEGER,                -- null if unavailable, NOT zero
    url             TEXT,                    -- link to original record
    domain          VARCHAR(100),            -- search domain used to retrieve it
    ingested_at     TIMESTAMP DEFAULT NOW(),

    -- Prevent duplicate imports from the same source
    CONSTRAINT uq_source_document UNIQUE (source, source_id)
);

-- Index for filtering by source and year
CREATE INDEX idx_documents_source_year ON documents(source, year);
CREATE INDEX idx_documents_domain ON documents(domain);

-- ============================================================
-- 2. technologies — Topics discovered by NLP clustering
-- ============================================================
-- Each row is one technology topic from KMeans clustering.
-- top_terms are the most representative TF-IDF terms for the cluster.
CREATE TABLE technologies (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255)  NOT NULL,  -- readable name (manually refined)
    cluster_id      INTEGER       NOT NULL,  -- KMeans cluster number
    top_terms       TEXT,                    -- comma-separated top TF-IDF terms
    document_count  INTEGER DEFAULT 0,       -- number of documents in this topic
    description     TEXT,                    -- optional manual description
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- 3. document_topics — Links documents to technology topics
-- ============================================================
-- A document belongs to exactly one primary topic (KMeans assigns one cluster).
-- We store the assignment so we can trace which documents support each topic.
CREATE TABLE document_topics (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    technology_id   INTEGER NOT NULL REFERENCES technologies(id) ON DELETE CASCADE,

    CONSTRAINT uq_document_topic UNIQUE (document_id, technology_id)
);

CREATE INDEX idx_document_topics_tech ON document_topics(technology_id);

-- ============================================================
-- 4. technology_year_indicators — Yearly counts per topic per source
-- ============================================================
-- Stores patent_count and paper_count separately so we never
-- confuse patents with papers at the data level.
CREATE TABLE technology_year_indicators (
    id              SERIAL PRIMARY KEY,
    technology_id   INTEGER NOT NULL REFERENCES technologies(id) ON DELETE CASCADE,
    year            INTEGER NOT NULL,
    patent_count    INTEGER DEFAULT 0,
    paper_count     INTEGER DEFAULT 0,
    total_count     INTEGER DEFAULT 0,
    avg_citations   REAL,                   -- average citations that year, null if unavailable
    is_complete     BOOLEAN DEFAULT TRUE,   -- false if year may have incomplete data

    CONSTRAINT uq_tech_year UNIQUE (technology_id, year)
);

CREATE INDEX idx_indicators_tech_year ON technology_year_indicators(technology_id, year);

-- ============================================================
-- 5. trend_scores — Computed Trend Score for each technology
-- ============================================================
-- One row per technology. Stores the overall score, each component,
-- the label, and the configuration used to compute it.
CREATE TABLE trend_scores (
    id              SERIAL PRIMARY KEY,
    technology_id   INTEGER NOT NULL REFERENCES technologies(id) ON DELETE CASCADE,
    score           REAL    NOT NULL,        -- 0 to 100
    label           VARCHAR(50) NOT NULL,    -- Emerging, Growing, Stable, Declining, Unusual Spike
    growth_rate     REAL,                    -- component score (0-100)
    consistency     REAL,
    acceleration    REAL,
    activity_volume REAL,
    citation_signal REAL,
    lookback_years  INTEGER,                -- how many years were actually used
    min_activity_met BOOLEAN DEFAULT TRUE,  -- did the topic meet the minimum threshold?
    computed_at     TIMESTAMP DEFAULT NOW(),
    config_snapshot TEXT,                    -- JSON string of weights/thresholds used

    CONSTRAINT uq_trend_tech UNIQUE (technology_id)
);

-- ============================================================
-- 6. forecast_runs — Metadata about each forecasting run
-- ============================================================
-- Tracks which model was used, what data it trained on, and its metrics.
CREATE TABLE forecast_runs (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,  -- 'moving_average', 'linear_regression', 'xgboost'
    parameters      TEXT,                   -- JSON: window size, features, etc.
    train_years     VARCHAR(100),           -- e.g. '2012-2021'
    test_years      VARCHAR(100),           -- e.g. '2022-2024'
    mae             REAL,
    rmse            REAL,
    series_type     VARCHAR(50),            -- 'patent_count' or 'paper_count'
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- 7. forecasts — Individual forecast predictions
-- ============================================================
-- Each row is one predicted value for one technology, year, and series.
CREATE TABLE forecasts (
    id              SERIAL PRIMARY KEY,
    forecast_run_id INTEGER NOT NULL REFERENCES forecast_runs(id) ON DELETE CASCADE,
    technology_id   INTEGER NOT NULL REFERENCES technologies(id) ON DELETE CASCADE,
    year            INTEGER NOT NULL,
    series_type     VARCHAR(50) NOT NULL,   -- 'patent_count' or 'paper_count'
    predicted_value REAL    NOT NULL,
    lower_bound     REAL,                   -- simple uncertainty band (if supported)
    upper_bound     REAL,
    is_future       BOOLEAN DEFAULT TRUE,   -- true=forecast, false=test-set prediction

    CONSTRAINT uq_forecast_point UNIQUE (forecast_run_id, technology_id, year, series_type)
);

CREATE INDEX idx_forecasts_tech ON forecasts(technology_id, year);

-- ============================================================
-- 8. ingestion_log — Record of each data collection run
-- ============================================================
-- Tracks what we downloaded, when, and whether it succeeded.
CREATE TABLE ingestion_log (
    id              SERIAL PRIMARY KEY,
    source          VARCHAR(50)  NOT NULL,  -- 'openalex' or 'uspto'
    domain          VARCHAR(100) NOT NULL,  -- search domain name
    started_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMP,
    status          VARCHAR(20)  NOT NULL DEFAULT 'running', -- running, success, failed
    records_fetched INTEGER DEFAULT 0,
    records_saved   INTEGER DEFAULT 0,
    error_message   TEXT,
    search_terms    TEXT                    -- the terms used for this run
);
