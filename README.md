# remote-boundless-DAS
This is a remote implementation of boundless DAS, modified from the official
`nnsight` local implementation of boundless DAS tutorial: https://nnsight.net/notebooks/tutorials/boundless_DAS/.

It expects a folder in the same working directory called `counterfactuals` 
containing CSV files with counterfactuals named `train.csv`, `val.csv`, and `test.csv`.
Each CSV should have the following three columns (they can have more):
    - "base_prompt": a base prompt string
    - "source_prompt": a source prompt string
    - "cf_label": a label string, either "Yes" or "No"