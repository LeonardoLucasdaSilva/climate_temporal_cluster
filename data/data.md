# Raw data

- `inmet_daily/` contains the existing station-level daily CSV hierarchy.
- `inmet_hourly/` is reserved for hourly INMET files and intentionally remains
  separate so loaders cannot silently mix temporal resolutions.

The configured roots are `INMET_DAILY_ROOT` and `INMET_HOURLY_ROOT` in
`src/config.py`. `DATA_ROOT` remains a compatibility alias for the daily root.
