python run.py   --input data/ml-1m-samples.json   \
--output outputs/result.json   --dataset ml-1m   --model ./models/llama-3.1-8b-instruct   \
--nodes diversity,accuracy,fairness,stop   --user-features user_id,age,occupation  \
--history-features item_id,title,genre,year   --item-features item_id,title,genre,year,score   \
--focus "Overall Performance"   --history-max 20   --max-count 3   \
--top-k 10   --temperature 0.0   --max-tokens 2048   --max-retries 5