# Model

This folder contains the generic models used in subsequent simulations. The complex model comes from the work of [Rajagopal et al. (2016) and Uhlrich et al. (2022)](https://simtk.org/projects/fbmodpassivecal), while the simple model comes from [Denton & Umberger (2023)](https://doi.org/10.1002/cnm.3777). If using these models, please cite the appropriate works:

**Uhlrich, S.D., Jackson, R.W., Seth, A. Kolesar, J.A., Delp, S.L. Muscle coordination retraining inspired by musculoskeletal simulations reduces knee contact force. *Sci Rep*, 12, 9842 (2022). doi: [10.1038/s41598-022-13386-9](https://doi.org/10.1038/s41598-022-13386-9)**

**Denton, A.N., Umberger, B.R. Computational performance of musculoskeletal simulation in OpenSim Moco using parallel computing. *Int J Numer Meth Biomed Engng*, 39, e3777 (2023). doi: [10.1002/cnm.3777](https://doi.org/10.1002/cnm.3777)**

### Model Edits

Each model was adapted to be: (i) lower limb only, which involved removing the torso and upper limb bodies while adding the mass of these removed bodies to the pelvis; and (ii) adding a marker set that matches the experimental dataset. 

**Simple Model**

The simple model came with contact geometry to simulate foot-ground contact, which was unnecessary for this work and hence was removed. To ensure it could align with the motion data from the 3D model, the planar nature of the model was expanded to include hip adduction and rotation, and a 6-DoF pelvis connection to the ground. A patellofemoral joint <u>was not</u> added to this simple model.

**Complex Model**

During the scaling step, the muscles of the complex model are altered to the [DeGrooteFregly2016Muscle](https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1DeGrooteFregly2016Muscle.html) to align with the simple model and also be used within OpenSim Moco simulations.