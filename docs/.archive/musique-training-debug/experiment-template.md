# Experiment Result Template

Use this template for documenting each experiment result.

## Experiment

### Setup
**Command**:
```bash
[Exact command used]
```

**Key Parameters**:
- Environment:
- Dataset: 
- Model:
- Batch size:
- Special flags:

### Hypothesis Prediction
**Expected if Validated**: [What would confirm the hypothesis]  
**Expected if Invalidated**: [What would rule it out]

### Results

#### Training Metrics
- **Reward Trend**: [Stable/Improving/Declining/Oscillating]
- **Final Reward**: [X.XX compared to baseline Y.YY]
- **Training Stability**: [Stable/Unstable/Gradient explosion/etc.]

#### Diagnostic Metrics (if applicable)
- **Advantage Statistics**: 
- **Context Length Stats**:
- **Token Masking Stats**:
- **Other Relevant Logs**:

#### Key Observations
- [Important patterns noticed]
- [Unexpected behaviors]
- [Comparison to previous experiments]

### Conclusion

**Hypothesis Status**: ✅ Validated / ❌ Invalidated / 🤔 Inconclusive

**Confidence Level**: High / Medium / Low

**Reasoning**: 
[Explain why you reached this conclusion based on the evidence]

**Next Steps**:
- [ ] [What to do based on these results]
- [ ] [Which hypothesis to test next]
- [ ] [Any follow-up needed]

### Files/Logs
- **WandB Run**: [link if available]
- **Log Files**: [paths to relevant logs]
- **Model Checkpoints**: [if saved]

---

**Notes**: [Any additional context or observations]