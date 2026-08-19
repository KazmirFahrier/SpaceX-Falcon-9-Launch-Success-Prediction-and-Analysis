# Data and notebook provenance

The runnable package uses the processed Falcon 9 launch dataset published for the IBM Data
Science Capstone course. The application downloads that dataset from the IBM Cloud Object
Storage URL declared in `src/spacex_falcon/data.py` and caches it locally. Generated data files
are excluded from version control.

The root level notebooks originated as IBM Skills Network course exercises and retain their
original author, copyright, and source notices. They are preserved as historical learning
artifacts. They are not the implementation used by the packaged dashboard or its tests.

The repository does not currently declare a software license. Anyone publishing or reusing the
course notebooks should first confirm the applicable IBM Skills Network terms. A repository
owner should choose a license for newly authored package code as a separate, explicit decision.
