"""
LemGendary Dataset Modernization package.

Implementation of the two modernization tools; the public entry points stay
in their original modules, which re-export from here:

  - tools.modernization_queue  batch queue (zip / loose / MDS -> containers,
                               Kaggle publication)
  - tools.modernize_manifold   legacy ``...Large`` manifold -> modern manifold

Modules:
  - common            shared image/YAML/cleanup helpers
  - queue_items       ManifoldQueueItem, registry + filesystem scan, status labels
  - loose_convert     loose images/targets directory -> WebDataset / MDS shards
  - mds_transcode     legacy MDS shards -> WebDataset shards
  - kaggle_push       publish a containerized manifold to Kaggle
  - queue_options     queue run option objects
  - queue_runner      queue ordering, planning, conversion dispatch, run loop
  - manifold_scan     registry I/O and eligible legacy manifold discovery
  - manifold_select   eligible manifold selection and confirmation
  - manifold_build    copy + transcode of a legacy manifold into a new folder
  - manifold_publish  docs, container conversion, Kaggle re-upload, reporting
  - manifold_run      ``modernize_manifold`` run phases
"""
