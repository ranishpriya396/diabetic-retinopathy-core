diabetic-retinopathy-core/
│
├── storage/                # Local emulation of S3 object storage buckets
│   ├── raw/                # Holds original incoming patient fundus images
│   └── heatmaps/           # Holds generated Grad-CAM XAI output overlays
│
├── database.py             # SQLAlchemy ORM database initialization engine
├── models.py               # Database relational tables configuration schema
├── main.py                 # Core FastAPI Microservice Engine Architecture
└── requirements.txt        # Updated full-stack dependency manifest
