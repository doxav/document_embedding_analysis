# BigSurvey Cache

This directory is a local cache for BigSurvey archives and extracted dataframe files.
Its large payloads are ignored by git.

Recreate the default 20-item MQS evaluation dataset:

```bash
python scripts/import_bigsurvey.py --n 20 --split test --skip-embeddings
```

This downloads `split_survey_df.tar.gz` from the configured Google Drive alias if it
is missing, extracts it under `data/bigsurvey/extracted_split/`, and writes output to
`output/bigsurvey/`.

To check the original archive instead, write it outside the canonical output
directory:

```bash
python scripts/import_bigsurvey.py \
  --archive original \
  --n 20 \
  --split test \
  --output-dir /tmp/dea_bigsurvey_original_check \
  --skip-embeddings
```

The current original archive test split yields 18 usable examples with `--n 20`;
rows without required cited-paper abstracts are skipped.

If network access to Google Drive is unavailable, manually place one of these files
in this directory and rerun the importer:

- `split_survey_df.tar.gz`
- `original_survey_df.tar.gz`

Do not commit downloaded archives or extracted dataframe directories.
