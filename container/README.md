# Description
Definition of singularity image for the FEniCS project.

## Usage
To build an image:
`singularity build sik.simg sik.def` 

When scheduling a job on cluster:
`singularity exec sik.sif echo python --version`
