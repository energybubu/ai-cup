# ai-cup
## .env sample (put `.env` file under the project root directory)
```shell
OPENAI_API_KEY = "your-openai-api-key"
```
## Run
```shell
python main.py \
  --model qwen_gpt \
  --output_path "pred.json" \
  --question_path "questions_example.json" \
  --source_path "datasets/reference/" \
  --use_cache
```
## File Structure
```
.
├ Preprocess
│ ├ preprocess.py       # main preprocessing
│ ├ translate.py        # text translation
│ └ io.py               # document io
├ Model
│ ├ constant.py         # model constants
│ ├ qwen.py             # qwen reranker
│ ├ gpt.py              # gpt utility
│ └ qwen_gpt.py         # qwen_gpt reranker
├ preds                 # prediction files
│ └ ...
├ debug_dataset         # for debugging
│ └ ...
├ main.py               # main program
├ requirements.txt
├ evaluate.py           # evaluation
├ .env                  # credentials
└ README.md
```