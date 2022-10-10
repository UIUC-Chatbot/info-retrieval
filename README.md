# info-retrieval
Implementing information retrieval models to aid in question-answering chatbots. 

## TODO

- [ ] Try this: https://github.com/facebookresearch/contriever
Specifically this variant: `mcontriever-msmarco: mcontriever with fine-tuning on MSMARCO.`

```python
# from their repo. Better yet, we should use huggingface
from src.contriever import Contriever
mcontriever_msmarco = Contriever.from_pretrained("facebook/mcontriever-msmarco")
```

- [ ] Come up with better questions for testing sentence and paragraph level embeddings
