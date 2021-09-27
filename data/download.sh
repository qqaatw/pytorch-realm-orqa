#!/bin/sh
gsutil -m cp -r \
  "gs://realm-data/cc_news_pretrained/" \
  "gs://realm-data/orqa_nq_model_from_realm" \
  "gs://realm-data/orqa_wq_model_from_realm" \
  "gs://orqa-data/enwiki-20181220/" \
  .