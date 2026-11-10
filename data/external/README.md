# External data

## `pfas_sites.xlsx`: PFAS Contamination Site Tracker (May 2024)

Compiled by the **PFAS Project Lab, Social Science Environmental Health
Research Institute (SSEHRI), Northeastern University**. It records known PFAS
contamination sites in the U.S. with location, sample matrix, sample date, and
measured levels, extracted from government websites, news articles, and public
documents.

- Included in this repository with the PFAS Project Lab's permission.
- Credit: PFAS Project Lab, Social Science Environmental Health Research
  Institute, Northeastern University.
- Under the tracker's terms, any public use of results derived from it must
  credit the lab and be shared with the lab team (pfasproject@gmail.com) at
  least 7 days before release, and the data may not be used in legal
  proceedings.
- The lab notes the tracker is a subset of all contamination sites and may be
  incomplete or out of date. Report inaccuracies to the lab.

`pipeline.py exposure` reads the `USA Contamination Sites` sheet. To use a
newer release, replace this file or set the `PFAS_FILE` environment variable.
