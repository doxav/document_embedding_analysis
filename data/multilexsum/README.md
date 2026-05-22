# MultiLexSum Cache

This directory is a local cache for official MultiLexSum release files. The large
payloads are ignored by git.

Recreate the default 20-item MQS evaluation dataset:

```bash
python scripts/import_multilexsum.py --n 20 --split test --skip-embeddings
```

Official mode downloads these files if they are missing:

- `{split}.json`, for example `test.json`
- `sources.json`

`sources.json` is large, so the importer memory-maps it and copies only the source
documents needed for the selected records into `output/multilexsum/`.

Offline/local mode is available when the files are already present:

```bash
python scripts/import_multilexsum.py \
  --source local \
  --records-json data/multilexsum/test.json \
  --sources-json data/multilexsum/sources.json \
  --n 20 \
  --output-dir /tmp/dea_multilexsum_local_check \
  --skip-embeddings
```

For official split files, the importer treats the file name as the split. It does
not treat the record-level `subset` field as a train/dev/test selector.

If network access to Hugging Face and the AI2 S3 mirror is unavailable, manually place
the official split file and `sources.json` in this directory and rerun the importer.

Do not commit official JSON payloads.
