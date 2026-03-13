# Analysis Scripts

Evaluation-only scripts for the AutoReview benchmark paper. These are NOT part of the core pipeline.

## Bibliography Injection

Injects a human review's bibliography into a KnowledgeBase snapshot for the retrieval-controlled experimental condition.

### Usage

```bash
# Step 1: Inject bibliography from reference PDF
python paper/analysis/inject_bibliography.py \
    --pdf paper/references/car_t_resistance_2019.pdf \
    --topic "CAR-T therapy resistance mechanisms" \
    --domain biomedical \
    --output paper/snapshots/car_t_injected.json \
    --cache paper/cache/resolution_cache.json

# Step 2: Resume pipeline from full_text_retrieval
autoreview resume paper/snapshots/car_t_injected.json \
    --start-from full_text_retrieval \
    --model claude-sonnet-4-6
```

### What it does

1. Extracts bibliography from a reference review PDF
2. Parses each reference line to extract DOIs and approximate titles
3. Resolves references to full paper records via Semantic Scholar and OpenAlex
4. Builds a pre-populated KnowledgeBase with all resolved papers auto-screened
5. Saves as a snapshot JSON that can be resumed with `autoreview resume`

### Resolution confidence tiers

- **High**: DOI exact match
- **Medium**: Title fuzzy match >= 85%
- **Low**: Title fuzzy match 70-85%

Unresolvable references are logged and excluded. The exclusion rate is reported as a metric.
