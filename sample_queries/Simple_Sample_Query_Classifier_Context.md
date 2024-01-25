********************************************
Launch Model in Classifier COntext
********************************************

make run CONTEXT_TYPE=classifier

********************************************
PROMPT EXAMPLE IN Classifier CONTEXT
********************************************

curl -X POST -H "Content-Type: application/json" --no-buffer 'http://127.0.0.1:3030/token_stream' -d '{"query":"I like this tiny shirt." }'

********************************************
ANSWER
********************************************

Fashion

********************************************
PROMPT EXAMPLE IN Classifier CONTEXT
********************************************

curl -X POST -H "Content-Type: application/json" --no-buffer 'http://127.0.0.1:3030/token_stream' -d '{"query":"I like this phone" }'

********************************************
ANSWER
********************************************

Electronics

********************************************
PROMPT EXAMPLE IN Classifier CONTEXT
********************************************

curl -X POST -H "Content-Type: application/json" --no-buffer 'http://127.0.0.1:3030/token_stream' -d '{"query":"What is the weather like" }'

********************************************
ANSWER
********************************************

General

