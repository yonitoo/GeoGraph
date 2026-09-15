# ReasoningBG geography subset

This folder pulls in the geography-12th slice of the [`reasoning_bg`](https://github.com/mhardalov/bg-reason-BERT)
dataset (Hardalov, Koychev & Nakov, 2019) as an additional source of Bulgarian
matriculation-exam geography questions, and checks it for overlap against this
project's own benchmark before merging in any new questions.

- Source dataset: [mhardalov/bg-reason-BERT](https://github.com/mhardalov/bg-reason-BERT)
  ([paper](https://arxiv.org/abs/1908.01519))
- `extract_geography_questions.py` - downloads the geography-12th split from the source repo.
- `analyse_candidates.py` - inspects/annotates candidate questions.
- `find_overlaps.py` - deduplicates against `testset/zeroshot_kg_rag_bggpt_curated_exams.jsonl`
  and writes out only the genuinely new questions.
- `reasoning_bg.py` - the original HuggingFace `datasets` loader script for the source dataset,
  kept for reference.

If you use this data, please cite the original dataset:

```bibtex
@article{hardalov2019beyond,
  title={Beyond english-only reading comprehension: Experiments in zero-shot multilingual transfer for bulgarian},
  author={Hardalov, Momchil and Koychev, Ivan and Nakov, Preslav},
  journal={arXiv preprint arXiv:1908.01519},
  year={2019}
}
```
