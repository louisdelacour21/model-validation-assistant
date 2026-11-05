# Blind third-party defects

Protocol (not yet run): a third party writes a defect into a copy of a model, sends only the SHA-256 of the defect file and a one-line description of its family, then the toolkit is run unchanged. The file is revealed after the run and the hash is checked. Detected / missed counts are added to `bench.json` with a Wilson interval.
