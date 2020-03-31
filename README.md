# Novelty Detection in Multispectral Mastcam Images
## Motivation
Science teams for rover-based planetary exploration missions like the Mars Science Laboratory Curiosity rover have limited time for analyzing new data before making decisions about follow-up observations. There is a need for systems that can rapidly and intelligently extract information from planetary instrument datasets and focus attention on the most promising or novel observations. Several novelty detection methods have been explored in prior work for three-channel color images and non-image datasets, but few have considered multispectral or hyperspectral image datasets for the purpose of scientific discovery. The Mastcam instrument is a multispectral imaging system that acquires images of the Mars surface from the mast of the Mars Curiosity rover in visible to near infrared wavelengths for the purpose of scientific study. We performed a study to compare the performance of four novelty detection methods---Reed Xiaoli (RX) detectors, principal component analysis (PCA), autoencoders, and generative adversarial networks (GANs)---to identify novel geology in Mastcam multispectral images.

This repository exists to assist in experimenting with the novelty detecton codebase developed for this study through Jupyter notebooks in a Jupyter Lab. There is a Jupyter notebook for each of the four methods that demonstrates how to use each method for novelty detection and evaluate performance for the Mastcam dataset. To make dependency managing easy, everything is setup using Docker. 

Citation for this work: Kerner, H. R., Wagstaff, K. L., Bue, B. D., Wellington, D. F., Jacob, S., Horton, P., Bell III, J. F., Kwan, C., Ben Amor, H. (2020). Comparison of novelty detection methods for rover-based multispectral images. Under review.

Dataset: https://doi.org/10.5281/zenodo.1486195

## Getting Started
### Requirements
To setup  this you will need `docker` and `docker-compose` which can be found [here](https://docs.docker.com/compose/install/)!
All other dependencies will be installed through the Dockerfile. Be sure to have the docker daemon running!

### Pre-Docker Setup
Make a new directory for this system, enter it, and clone this repo
```
mkdir novelty_det
cd novelty_det
git clone (repo_url)
```

Make a new folder for data, enter the cloned repository and run `setup.py` to download the datasets
```
mkdir mcam_data
cd (repo_name)
python setup.py ../mcam_data
```
### Docker Setup

Edit the `.env` file specify which directorty contains the data
```
# Provide data sets
LOCAL_DATASETS=/path/to/mcam_data
```

**Optional**: Use the script `generate_token.py` to create a new Jupyter Lab password.
You will need this to login once the service is deployed.
Edit the `ACCESS_TOKEN` variable with the new SHA key.
**By default, the password is `asdf`.**

### Launching Docker
To launch docker, start from the repository's directory and start up the server
```
docker-compose up
```
Now you can open `https://localhost:8888` to access the Jupyter Lab! 
If prompted for a password use `asdf` of the password you set in the `.env` file.

In the Jupyter lab instance, this codebase will be mounted in `/home/jovyan/work` and the data will be in `/home/jovyan/data`.

To bring the server down simply use
```
docker-compose down
```
