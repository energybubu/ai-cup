# ai-cup
## Debug
```shell
python main.py \
  --model bm25 \
  --output_path debug_dataset/dataset/preliminary/pred.json \
  --question_path debug_dataset/dataset/preliminary/questions_debug.json \
  --source_path debug_dataset/deb_ref
```
## Run
```shell
python main.py \
  --model qwen_gpt \
  --output_path preds/qwen_gpt_chunking_pred.json \
  --question_path datasets/dataset/preliminary/questions_example.json \
  --source_path datasets/reference/ \
  --use_cache
```