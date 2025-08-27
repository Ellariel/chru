
def base_prompt(category_name, 
                description,
                key_features,):
                #keywords): # **Keywords:** “{keywords}”
    return f"""
You are tasked with applying qualitative codes to text excerpts, referencing their content. The purpose of this task is to determine whether each text excerpt represents the encoded phenomenon.

**Code Title:** “{category_name}”
**Description:** “{description}”
**Key Features:** “{key_features}“

When evaluating the text, you must:

1. Decide whether the code applies.
2. Provide a concise justification for your decision (2–3 sentences).
3. Indicate your confidence level (a number between 0.0 and 1.0).
4. List the applied code if relevant.

If the code **is applied**, format your answer as:

```
Justification: [2–3 sentence rationale for applying the code]
Confidence: [confidence score between 0.0 and 1.0]
Code Applied: [{category_name}]
```

If the code **is not applied**, format your answer as:

```
Justification: [2–3 sentence rationale for not applying the code]
Confidence: [confidence score between 0.0 and 1.0]
Code Applied: [None]
```

**Important:** Do not include any additional text after the “Code Applied:” line.
"""


def parse_output(output):
    try:
        if 'Justification:' in output:
            j = output.index('Justification:')
        else:
            j = 0
        if 'Confidence:' in output[j:]:
            c = output.index('Confidence:', j)
        else:
            c = 0
        if 'Code Applied:' in output[c:]:
            r = output.index('Code Applied:', c)
        else:
            c = 0
        if '{' in output:
            x = output.index('{', c)
        else:
            x = len(output)
        return {'justification' : output[j+14:c].replace('\n', ' ').replace('  ', ' ').strip().replace(';', ','),
                'confidence' : output[c+11:r].replace('\n', '').strip(),
                'code_applied' : output[r+14:x].replace('\n', ''),
                'output' : output.replace(';', ',')}
    except:
        return {}