## GDELT Monthly Processing Pipeline (`0_load_data.py`)

### Overview
The data ingestion layer is powered by `0_load_data.py`, an enterprise-grade, highly optimized data pipeline designed to ingest, clean, and compress massive amounts of **GDELT Global Knowledge Graph (GKG) V2.1** data. It downloads monthly subsets of both standard and translingual raw CSV ZIP files, processes them in parallel, maps high-cardinality string data (media sources) into numeric IDs, and writes the results into highly compressed, partition-friendly **Apache Parquet** files.

### Performance Architectures & Bottleneck Fixes
The pipeline utilizes a specialized multi-tier architecture to maximize system resource utilization ($I/O$, CPU, and Disk):

* **Network Concurrency (`ThreadPoolExecutor`):** Utilizes high concurrency (`net_workers=32`) to saturate network bandwidth during the file-download phase.
* **CPU Parsing (`ProcessPoolExecutor`):** Distributes heavy text-parsing tasks across multiple CPU cores (`cpu_workers=50`).
* **Arrow IPC Serialization (Zero-Copy):** Traditional `pandas` DataFrames containing heavy text are incredibly expensive to pass through Python's `pickle` serialization (adding ~7–8 seconds of IPC overhead per file). This script leverages **Apache Arrow IPC streams**, yielding native, zero-copy binary buffers that decrease overhead to a few milliseconds.
* **Asynchronous Writer Thread (Producer-Consumer):** Data parsing and Parquet writing are completely decoupled. While CPU workers parse incoming chunks, a dedicated background thread accumulates tables and performs massive row-group flushes dynamically.

### Key Features
* **Dual Master List Ingestion:** Automatically crawls both the standard GDELT master file list and the translingual (translated) data streams.
* **Source Mapping Dictionary:** Converts resource-heavy string fields like `SourceCommonName` into compact `int32` identifiers (`SourceCommonName_ID`), saving massive amounts of disk space and memory. Mappings are persisted locally in `gdelt_sources_mapping.json`.
* **Memory-Efficient Row-Grouping:** Dynamically concatenates processed records and flushes them only when hitting the optimal row threshold (`row_group_size=500,000`).
* **Zstandard Compression:** Uses Parquet-native `zstd` compression with configurable levels to achieve high data density.
* **Automatic Buffer Purge:** Deletes processed ZIP files at the end of each monthly iteration to maintain a low local disk footprint.

### Extracted Schema Details
The pipeline extracts a targeted subset of crucial GKG fields, heavily optimizing their data types for subsequent downstream analytics:

| Column Name | Data Type | Notes / Optimization |
| :--- | :--- | :--- |
| `GKGRECORDID` | `string` | Unique identifier (Primary Key, duplicate-stripped) |
| `DATE` | `string` | Event timestamp string |
| `IsTranslingual` | `int8` | Flag (`1` if originating from the translation stream, `0` otherwise) |
| `SourceCollectionIdentifier` | `int8` | Numeric code indicating source type |
| `SourceCommonName_ID` | `int32` | Mapped integer key pointing to the media source dictionary |
| `DocumentIdentifier` | `string` | Target source URL or identifier |
| `EnhancedThemes` | `string` | Semicolon-delimited GDELT themes list |
| `EnhancedLocations` | `string` | Parsed locations mentioned in the text |
| `Persons` | `string` | Mentioned people |
| `Organizations` | `string` | Mentioned organizations |
| `Tone` | `float16` | Document sentiment score (extracted from `Tone_Raw` index 0) |
| `WordCount` | `int32` | Total words in document (extracted from `Tone_Raw` index 6) |
| `TranslationInfo` | `string` | Provenance details for translated documents |

### Operational Parameters & Run Command

#### Configuration
Fine-tune the processing behavior directly inside the script block initialization:

```python
pipeline = GDELTRollingPipeline(
    temp_dir='/data/gdelt/gdelt_buffer_temp',   # Ephemeral directory for downloaded ZIPs
    final_dir='/data/gdelt/gdelt_parquet_dbv2',  # Target directory for output Parquet files
    net_workers=32,                              # Max concurrent downloads
    cpu_workers=50,                              # Max parallel CPU processing cores
    zstd_compression_level=6,                    # Compression strength (1-22)
    row_group_size=500_000,                      # Target chunk size per Parquet row group
    write_queue_maxsize=30,                      # Depth limit for memory backup safety
)