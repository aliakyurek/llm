# LLM
Various LLM examples
* News_Article_Summarizer:
  * Summarizes an article on a URL-linked news site using OpenAI and Langchain.
  * Used API&Libraries: openai langchain newspaper3k
* Titanic_Data_analysis:
  * Analyzes Titanic survival dataset, by chatting with data, creating visualizations using OpenAI and Langchain CSV agent.
  * Finally, tries to create a random forest model and predict the test set by chatting with CSV agent.
  * Used API&Libraries: openai langchain
* Llama2_7b_GPTQ:
  * Utilizes 4bit quantatized version of Llama2 7b Chat. "TheBloke/Llama-2-7b-Chat-GPTQ"
  * Used API&Libraries: transformers auto_gptq bitsandbytes
* Distilbert_Masking:
  * Utilizes DistilBertForMaskedLM from HF and finds masked word
  * Used API&Libraries: transformers
PEFT_finetune_Bloom7B_sentiment:
  * Finetunes "bigscience/bloom-7b1" for sentiment generatin
  * Used API&Libraries: bitsandbytes datasets accelerate loralib transformers peft
