# SMOL SQL

In this repository I will apply what I learned on [SMOLCourse](https://github.com/huggingface/smol-course/tree/main) to
to develop a model to generate SQL queries. Here is the HF Page: <https://huggingface.co/nahumsa/SmolLM2-FT-SQL>

## 2025-01-07

I was able to train the model using a CPU (AMD Ryzen 7 5700G) + 16GB of RAM. The dataset that I use to train is a SQL dataset by
`gretalai`: [gretelai/synthetic_text_to_sql](https://gretel.ai/blog/synthetic-text-to-sql-dataset). Where I use the following template:

```python
[
    {"role": "user", "content": sample["sql_prompt"]},
    {"role": "assistant", "content": sample["sql"]},
]
```

However when I finish training, the model keeps generating the text without adding a stop sequence, e.g.:

```bash
Inference:
<|im_start|>user
What are the distinct entries from city?<|im_end|>
<|im_start|>assistant
SELECT city, EXTRACT(YEAR FROM entry_date) AS year FROM entries GROUP BY city; SELECT city, EXTRACT(YEAR FROM entry_date) AS year FROM entries GROUP BY city; SELECT city, EXTRACT(YEAR FROM entry_date) AS year FROM entries GROUP BY city; SELECT city, EXTRACT(YEAR FROM entry_date) AS year FROM entries GROUP BY city; SELECT city, EXTRACT(YEAR FROM entry_date
```

In order to fix that I will add a specific stop sequence when fine tuning the model.
