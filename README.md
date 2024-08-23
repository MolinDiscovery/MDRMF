# MDRMF
Multiple drug resistance machine fishing
- A project to explore ML models using active pool-based learning (machine fishing) for transporters implicated in MDR.

# Installation
python -m pip install MDRMF
[MDRMF](https://pypi.org/project/MDRMF/)

# Why is it called MDRMF?
MDRMF is a python package that was developed as part a project to find inhibitors of ABC transporters which cause multiple-drug resistance towards chemotherapeutics. The machine fishing part refers to the fact that we try to catch candidate drugs from a pool of data.

# What does MDRMF do?
MDRMF is a platform that can help find candidate drugs for a particular disease target. The software have two modes. 
1) A retrospective mode for testing and optimization purposes of the active learning workflow.
2) A prospective mode for usage in experimental settings.

**Retrospective part**
This is for testing and optimization purposes. You have dataset with SMILESs that is fully labelled with some score (e.g. docking). The software can evaluate how many hits its able to obtain with the specified settings.

**Prospective idea**
The software was designed to be used on experimental data. That is, you have a list of SMILESs from the dataset you're investigating. You select X number of molecules from the dataset and test them to obtain labels. These labels are assigned to the corresponding molecules and in given to the software for training. The software will then return X number of molecules that it wants you to test next.