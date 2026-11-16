#!/usr/bin/env bash
# COCO val2014 images + instance annotations and the CHAIR synonym list.
# POPE and HallusionBench are downloaded automatically from the Hugging Face Hub.
set -euo pipefail
DATA=${DATA:-data/coco}
mkdir -p "$DATA/annotations"
cd "$DATA"
if [ ! -d val2014 ]; then
  curl -L -o val2014.zip http://images.cocodataset.org/zips/val2014.zip
  unzip -q val2014.zip && rm val2014.zip
fi
if [ ! -f annotations/instances_val2014.json ]; then
  curl -L -o annotations_trainval2014.zip http://images.cocodataset.org/annotations/annotations_trainval2014.zip
  unzip -q -j annotations_trainval2014.zip annotations/instances_val2014.json -d annotations
  rm annotations_trainval2014.zip
fi
if [ ! -f synonyms.txt ]; then
  curl -L -o synonyms.txt https://raw.githubusercontent.com/LisaAnne/Hallucination/master/data/synonyms.txt
fi
echo "COCO data ready in $DATA"
