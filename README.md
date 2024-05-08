# Mapping between pipeline logic units and script implementations

### Preface
Most of these scripts rely on the existence of a configuration file with this structure:
`````
dfsdijfosi
`````
The file path of the configuration file must be passed as an input parameter (`--conf`).

Additionally, most of the scripts rely on the existence of a PostgresDB.

A further dependency of some scripts is introduced by the use of VPNs, which are necessary to avoid traffic filtering/limiting policies from hosting providers when the volume of requests is too high. 

## Step 1

### PDF Metadata Analysis
- __PDF_graph/*__: the scripts in this folder run an existing installation of PeePDF, extract clickable URLs by parsing the PDF tree, and finally perform a series of operations producing useful metadata (e.g., extraction of first-level domain, netloc, string matching for IoC, etc.). The entry point is `extract_all_urls.py`.
   
### PDF Status Check
- __pdf-check-pipeline/*__: this is a standalone module that can be run on its own. The modules should be run in the following order:
    - activate venv and set as python environment path that of the pdf-check-pipeline folder
    - `read_urls.py`
    - `test_vpns.py`
    - `process_urls.py`
    - `create_daily_backup.py`
    - `save_results.py`

## Step 2
### URL Analysis Module
- __url-analysis-module/host_info_gathering.py__: collects DNS and WHOIS information of the domains extracted on the day.

## Step 3
### Access Control Lists
- __access-control-lists/*__: this is a standalone module which takes as input a S3-compatible bucket URL and scans for misconfigured ACLs. Entry point: `wrapper_s3scanner.py`.

### Software, Version
- __software_version/*__:
    - Collecting websites that need to be inspected: `extract_20_urls.py` and `assign_wpta_urls.py` check which domains still do not have any IoC indicator and schedule them for inspection by the WPTAgent component.
    - A separate and independent instance of WPTAgent runs daily and inserts raw scan results in the DB.
    - `fill_support_cf_tables.py` and `fill_support_cf_random_table.py` process the raw scan result data and extract clean software, version information, storing them in another table
    - `NVD_lookup.py` and `wpscanDB_lookup.py` select the clean software, version information from the DB and check whether there is any corresponding vulnerability entry in the DB.
    - fill_support
    - extract 20
