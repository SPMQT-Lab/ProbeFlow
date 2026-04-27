# ProbeFlow Roadmap

Below is my idea for the roadmap of ProbeFlow. Please correct it anywhere you see fit. The impression of confidence you may get in my writing is entirely false haha. 

This is not meant to impose a design. I wrote it because I think having a shared architecture would make it easier for me and others to contribute without duplicating work or adding low-priority features.

however, i am excited for you to read about my image-history idea. I think it provides a very useful and key differentiator, and also easily gives the foundational architecture for the entire project. 

## Vision

ProbeFlow should become the central workflow manager for STM image analysis.
It should not try to replace Gwyddion, Fiji, ImageJ, or AngstromPro.

The goal is to make ProbeFlow the place where STM workflows are organized, tracked,
extended, and eventually automated.

ProbeFlow should be the home base. External tools should be excursions from ProbeFlow, and their outputs should return
to ProbeFlow as tracked, reproducible derived results. Exact reproducibility is impossible in the case that one makes an excursion for a particular data analytic tool, however, this will become less and less necessary as the library of ProbeFlow compatible tools increases, in this case, the exact image operation can and will be tracked, giving 100% accurate reproducibility. 

Anyway, a large part of ProbeFlow's vision derives from its ability to quickly and easily adapt algorithms on github to become ProbeFlow plug-ins. This is central to ProbeFlow. With this feature, we do not *have* to manually write and check a variety of image processes, we can simply create a plug-in adaptor from Fiji, to ProbeFlow, and then we immediately have access to an immense library of plug-in's from imajeJ. 

---

## The Wedge

ProbeFlow should differentiate through:

- easy installation
- a rigorous processing history model
- Python API
- CLI
- batch processing
- a simple and stable plugin template
- Createc and Nanonis support
- external-tool integration with history tracking.

The point is not to out-image-process Gwyddion or Fiji.
The point is to be the workflow layer that ties them together. 

---

## Core Design

The GUI is difficult. But the backend comes first. To rigorously define the backend we rigorously define the building blocks of ProbeFlow. Its most difficult element, is its tracking, whether that be of previous image operations, or of measurements / results. These are all non-trivial things to track and can easily get messy fast in an exponentially growing code base. Batch processing is easy, since it is only sequentially applying backend rules, and image galleries are a GUI problem. The most important thing that we get right for easy plug-in developmenet and a well prepared project for later machine learned training is storing the data efficiently and logically. So here I try and define the exact objects of ProbeFlow that should be rigorously defined. 

An image is fundamentally a vector. More precisely, an STM image can be treated as an element of some finite-dimensional vector space, for example:

image \in R^(Ny × Nx)

There are then two fundamental things one can do to this vector.

The first we call a transformation. A transformation takes one or more image vectors (or measurement vectors, covered later) and produces one or more new image vectors.

Examples:

- plane subtraction
- line correction
- smoothing
- cropping
- segmentation into sub-images
- external editing in Gwyddion or Fiji
- registration
- averaging
- subtraction between images

So a transformation is generally of the form:

V^n → V^m

If an operation produces several images, it is still a transformation. For example, segmentation is not a separate fundamental class of operation. It is simply a transformation that takes one image and produces multiple image objects.

The second is a measurement. A measurement takes one or more image vectors, and possibly previous measurements, and produces a measurement result, like projecting a vector onto another. 

Examples:

- atom count
- lattice spacing
- line profile
- FFT peak positions
- terrace height
- comparison between previous measurements

So a measurement is generally of the form:

V^n → Result

or, in some cases:

Result^n → Result

or:

V^n × Result^m → Result

If an operation produces both a new image and a measurement, then it is both a transformation and a measurement. That is fine. It just means the operation must satisfy the characteristics of both classes: it must create a new image node and a new measurement node, both connected to the correct inputs.

There are also parsers and writers, but these are boundary operations rather than the central scientific operations.

The foundational object is the scan.

A scan consists of:

- a raw source file
- a header / metadata object
- N image objects
- usually FT, FC, BT, BC
- each image stored as a Numpy array
- scan-level metadata
- all attached image and measurement nodes (defined later)

The raw scan file should be immutable. Every image object derived from it remains connected back to this original raw scan.

The image objects are nodes on a provenance graph.

The transformations are lines that connect any number of image nodes / measurement nodes, to an image node.

The measurements are lines connecting any number of image nodes / measurement nodes, to a measurement node. 

Each image node should encode:

1. identity of the raw scan and channel (FT, FC, BT, BC) it ultimately derives from

2. identity of its input node(s) 

3. identity of the transformation that created it

4. exact transformation parameters

5. software / plugin version used for the transformation

6. warnings or assumptions made during the transformation

7. its units

8. A numpy array of the image data itself

This means that, in principle, one can reproduce an image node by applying a transformation with identical identity, version, parameters, and inputs to the previous node or nodes.

Each measurement node should encode:

0. identity of the raw scan it ultimately belongs to

1. identity of its input node or input nodes

2. identity of the measurement operation that created it

3. exact measurement parameters

4. software / plugin version used for the measurement

5. warnings or assumptions made during the measurement

6. the measurement value itself, or a reference to the saved measurement data

7. units, where applicable

8. The measurment itself. Whether that be a .png, a number, an array, whatever. 

This means that measurements do not get lost as disconnected CSV files or screenshots. They remain tied to the exact image node,  and scan from which they were produced.

A session is then defined as a collection of scans (like grouping scans by material or scanning time). Only scans reference the session. Nodes do not reference the session. This means that i can change the session a scan belongs to, and since nodes are only linked to the scan, the nodes will be attached with it. If the Nodes referenced the session, then we would have a large mess if we changed the session of a scan. 

The central architecture should be that a session owns a scans, and each scan owns a provenance graph defined by image nodes and measurement nodes, connected by transformation and measurement lines. 

In this graph:

- raw files are immutable roots
- parsers create initial scan objects
- scans contain image nodes
- transformations create new image nodes
- measurements create measurement nodes
- writers create artifact nodes
- all nodes remain tied back to the original raw scan

So ProbeFlow would not merely process STM images. It would track the full life of an STM image; what raw file it came from, what transformations were applied, what measurements were made, what outputs were exported, and how every derived object can be traced back to the original data.

---

## Plugin Development, and why this architecture makes it easy

This architecture makes plugin development simple because every plugin only has to answer: 

> What does this operation take in, and what does it return?

A plugin does not need to understand the whole ProbeFlow GUI, file system, session structure, or provenance mechanics.

It only needs to define whether it is a:

- transformation
- measurement
- writer
- parser

A transformation returns new image nodes.

A measurement returns measurement nodes.

A writer returns external artifacts.

A parser returns a scan.

This means a plugin can be described by only:

```python
Plugin:
    name
    version
    operation_type
    input_types
    output_types
    parameters
    function(inputs, parameters) -> outputs
```

## Conseuqences of this data structure

With this data structure, we have measurements and images tied to scans, and scans tied to sessions. So basic data selection can give us any subset of this session. Suppose the session was organised such that it contained all the scans of a tuesday worth of scanning copper (110). We can collect all of the atom counting results of our image processing of these scans by 
`running session.measurements.atom_counting`
We could also decide to collect all of the measurements done on a particular scan: `session.scan.measurements`. etc...


## Summary

Session owns scans.
Scan owns graph.
Image/Measurement/Operation nodes live inside the scan graph.
Plugins declare inputs and outputs.
ProbeFlow handles graph/provenance/storage.