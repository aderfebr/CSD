# File Structure
```
janus/
├── AR.py
├── CSD.py
├── GSD.py
├── JD.py
├── SJD.py
T2I.py
```

# Dependencies
```
pip install -r requirements.txt
```

# Run Example
```
python T2I.py
```

# Basic Configuration
```
# Modify conversation content
conversation = [
    {
        "role": "<|User|>",
        "content": "Your image description text",
    },
    {"role": "<|Assistant|>", "content": ""},
]
```

# Generation Modes

```
from janus.AR import generate
from janus.CSD import generate
from janus.GSD import generate
from janus.JD import generate
from janus.SJD import generate
```