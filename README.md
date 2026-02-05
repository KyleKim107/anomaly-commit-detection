## How to run

python3 isolation_forest.py <repo_name> <number of commits to check>

ex.
python3 isolation_forest.py pandas 1000

ensure the repo that you are testing is in the same folder as the script

flowchart LR
    subgraph Input
        A[📄 CVE Data]
    end
    
    subgraph Repo
        B{Exists?}
        C[📥 Clone]
        D[📂 Local]
    end
    
    subgraph Cache
        E{Cached?}
        F[📦 Load]
        K[💾 Save]
    end
    
    subgraph Process
        G[⛏️ Mine]
        H[🔍 Parse]
        I[📊 Features]
        J[🧠 Embed]
    end
    
    subgraph Detect
        L[🎯 Isolation Forest]
        M[📈 Evaluate]
        N[📋 Report]
    end
    
    A --> B
    B -->|No| C --> D
    B -->|Yes| D
    D --> E
    E -->|Yes| F --> L
    E -->|No| G --> H --> I --> J --> K --> L
    L --> M --> N
    
    style A fill:#e1f5fe
    style F fill:#fff3e0
    style K fill:#fff3e0
    style L fill:#f3e5f5
    style N fill:#e8f5e9