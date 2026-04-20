-- Migration 001 — Initial schema
-- Run via: supabase db push  OR  psql $DATABASE_URL -f this_file.sql
-- Wraps schema.sql in a transaction so it's atomic.

BEGIN;

-- \i ../schema.sql  -- DISABLED: \i is a psql-specific command and will fail in the Supabase SQL Editor.
                     -- (Assuming you have already run schema.sql directly as per the previous step).

-- Seed: product category lookup (heuristic from UCI stock codes)
CREATE TABLE IF NOT EXISTS product_category_map (
    stock_code_prefix   TEXT    PRIMARY KEY,
    category            TEXT    NOT NULL
);

INSERT INTO product_category_map (stock_code_prefix, category)
VALUES
    ('20',  'gift_wrap'),
    ('21',  'homewares'),
    ('22',  'kitchenware'),
    ('23',  'bags_and_cases'),
    ('47',  'stationery'),
    ('48',  'seasonal'),
    ('71',  'art_and_craft'),
    ('84',  'decorative'),
    ('85',  'novelty'),
    ('POST','postage'),
    ('DOT', 'dotcom'),
    ('C2',  'carriage'),
    ('M',   'manual')
ON CONFLICT DO NOTHING;

-- Amount bucket boundaries (filled after first data load via percentile calc)
CREATE TABLE IF NOT EXISTS amount_bucket_boundaries (
    bucket  SMALLINT PRIMARY KEY,    -- 1–5
    min_val NUMERIC(12, 4),
    max_val NUMERIC(12, 4),
    label   TEXT
);

COMMIT;