# Prime Streets Processor

A Python package to process real estate street price data, normalize prices, compute market factors, and generate CSV and PDF reports for top and bottom performing streets within each zone.

## Features

* Load and clean raw listing and cadastral data
* Normalize street names and zones
* Fuzzy match extracted street names to cadastral records
* Deflate prices by date and zone, and normalize price per m²
* Compute zone and street averages, factors, and market share
* Select top/bottom streets per zone based on configurable thresholds
* Export final results to CSV and PDF formats

## Prerequisites

* Python 3.8 or higher
* Dependencies listed in [requirements.txt](requirements.txt)

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd <repository_directory>

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from prime_streets_processor import PrimeStreetsProcessor

processor = PrimeStreetsProcessor(
    streets_csv="df_all_extracted_streets.csv",  # Extracted listings file
    catastro_csv="catastro_sophiq_20241210.csv",  # Cadastral reference data
    output_dir="outputs",                         # Directory for results
    min_ads=10,                                     # Minimum ads per street
    top_n=10                                        # Number of top/bottom streets per zone
)
processor.run()
```

## Configuration

* `streets_csv`: Path to the CSV of extracted street listings.
* `catastro_csv`: Path to the cadastre reference CSV.
* `output_dir`: Directory where CSV and PDF outputs will be saved.
* `min_ads`: Only include streets with at least this many ads in selection.
* `top_n`: How many top and bottom streets to select per zone.

## Outputs

* `prime_streets_top_final.csv`: Top streets in each zone by price factor.
* `prime_streets_bottom_final.csv`: Bottom streets in each zone by price factor.
* `prime_streets_top_final.pdf`: PDF report for top streets.
* `prime_streets_bottom_final.pdf`: PDF report for bottom streets.

## Environment Variables

Store sensitive credentials or settings in a `.env` file at the project root:

```
API_KEY=your_api_key_here
ANOTHER_SETTING=value
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
