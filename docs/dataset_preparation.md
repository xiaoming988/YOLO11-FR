# Dataset preparation

This repository does not redistribute the original datasets.

## GYU-DET-Crack

The main experiments use GYU-DET-Crack, which was extracted from the public GYU-DET bridge surface defect dataset. The original train, validation, and test partitions of GYU-DET were retained during subset construction. For each partition, only annotations belonging to the crack category were retained and remapped to a single crack class. Images containing at least one crack annotation were kept as positive samples, and a subset of non-crack images was retained as hard negative samples.

## Crack500

The additional benchmark experiment uses the converted Crack500 detection subset. Crack500 provides pixel-level crack masks. In this study, the masks were converted into bounding-box annotations by extracting connected crack regions and generating their minimum enclosing rectangles.

Please download GYU-DET and Crack500 from the original public sources cited in the manuscript.
